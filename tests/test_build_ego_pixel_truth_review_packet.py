from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from tools.build_ego_pixel_truth_review_packet import build_packet


class EgoPixelTruthReviewPacketTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.images = self.root / "images"
        self.images.mkdir()
        Image.new("RGB", (40, 40), "white").save(
            self.images / "camera0-frame0.png")

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def response(_image, _points, *, warnings=None):
        return {
            "parts": [{"points": [[10, 20], [30, 20], [30, 39], [10, 39]],
                       "holes": []}],
            "score": .8,
            "warnings": warnings or [],
        }

    def test_packet_writes_binary_candidate_and_never_claims_truth(self):
        output = self.root / "packet"
        packet = build_packet(
            self.images, output, self.response, cameras=(0,), frames=(0,))
        self.assertEqual(packet["sample_count"], 1)
        self.assertEqual(packet["candidate_available_count"], 1)
        self.assertFalse(packet["human_truth"])
        self.assertFalse(packet["human_confirmed"])
        self.assertTrue((output / packet["contact_sheet"]).is_file())
        self.assertEqual(len(packet["contact_sheet_sha256"]), 64)
        row = packet["samples"][0]
        self.assertFalse(row["eligible_for_candidate_merge"])
        self.assertNotIn("parts", row)
        mask = np.asarray(Image.open(output / row["candidate_mask"]))
        self.assertEqual(set(np.unique(mask)), {0, 255})

    def test_warning_is_exported_for_review_but_remains_blocked(self):
        output = self.root / "packet"
        packet = build_packet(
            self.images, output,
            lambda image, points: self.response(
                image, points, warnings=["positive point missed"]),
            cameras=(0,), frames=(0,))
        row = packet["samples"][0]
        self.assertTrue(row["candidate_observed"])
        self.assertFalse(row["candidate_available"])
        self.assertEqual(row["status"], "blocked_sam_warning")
        self.assertEqual(row["sam_warning_count"], 1)
        self.assertNotIn("warnings", row)

    def test_scope_and_existing_output_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "cameras"):
            build_packet(self.images, self.root / "bad", self.response,
                         cameras=(0, 0), frames=(0,))
        output = self.root / "packet"
        output.mkdir()
        with self.assertRaises(FileExistsError):
            build_packet(self.images, output, self.response,
                         cameras=(0,), frames=(0,))


if __name__ == "__main__":
    unittest.main()
