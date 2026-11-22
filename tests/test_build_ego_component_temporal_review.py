import unittest

import numpy as np
from PIL import Image

from tools.build_ego_component_temporal_review import (
    crop_box_around_bbox, parse_identity, photometric_evidence)


class EgoComponentTemporalReviewTests(unittest.TestCase):
    def test_parses_component_and_source_identities(self):
        component = parse_identity(
            "004-camera1-idx5-1793760663542839987-c002")
        source = parse_identity("004-camera1-idx22-1793760668642838987")
        self.assertEqual(component["package"], "004")
        self.assertEqual(component["camera"], "camera1")
        self.assertEqual(component["index"], 5)
        self.assertEqual(source["index"], 22)

    def test_centers_and_clamps_crop(self):
        self.assertEqual(
            crop_box_around_bbox([435, 398, 438, 401], (640, 512)),
            (388, 351, 484, 447))
        self.assertEqual(
            crop_box_around_bbox([1, 1, 3, 3], (640, 512)),
            (0, 0, 96, 96))

    def test_rejects_bad_identity_and_bbox(self):
        with self.assertRaisesRegex(ValueError, "unsupported"):
            parse_identity("bad-id")
        with self.assertRaisesRegex(ValueError, "outside"):
            crop_box_around_bbox([0, 0, 700, 2], (640, 512))

    def test_measures_dark_target_against_local_ring(self):
        image = Image.new("RGB", (7, 7), (200, 200, 200))
        image.putpixel((3, 3), (20, 20, 20))
        target = np.zeros((7, 7), dtype=bool)
        target[3, 3] = True
        evidence = photometric_evidence(image, target, radius=1)
        self.assertEqual(evidence["target_pixels"], 1)
        self.assertEqual(evidence["ring_pixels"], 8)
        self.assertLess(evidence["target_minus_ring_luma"], -170)

    def test_rejects_empty_photometric_target(self):
        with self.assertRaisesRegex(ValueError, "invalid"):
            photometric_evidence(
                Image.new("RGB", (4, 4)), np.zeros((4, 4), dtype=bool))


if __name__ == "__main__":
    unittest.main()
