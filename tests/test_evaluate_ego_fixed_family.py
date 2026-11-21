import io
import unittest

from PIL import Image

from tools.evaluate_ego_fixed_family import (
    derive_minimum_family,
    evaluate_fixed_family_image,
)


class EgoFixedFamilyTest(unittest.TestCase):
    def test_family_is_derived_from_development_only(self):
        attempts = [
            {"requested_positive_indices": [0, 2], "attempt_available": True,
             "covered_positive_indices": [0, 2]},
            {"requested_positive_indices": [1, 3], "attempt_available": True,
             "covered_positive_indices": [1, 3]},
            {"requested_positive_indices": [3], "attempt_available": True,
             "covered_positive_indices": [3]},
        ]
        discovery = {
            "schema_version": "labelagent.ego_subset_cover_ab.v1",
            "candidate_only": True,
            "human_truth": False,
            "development_frames": [0, 1],
            "held_out_frames": [2],
            "samples": [
                {"split": "development", "frame": 0, "attempts": attempts},
                {"split": "development", "frame": 1, "attempts": attempts},
                {"split": "held_out", "frame": 2, "attempts": []},
            ],
        }
        family = derive_minimum_family(discovery)
        self.assertEqual(family, ((0, 2), (1, 3)))

    @staticmethod
    def response(_image, points):
        parts = []
        for positive in [point for point in points if point["label"] == 1]:
            x, y = positive["x"], positive["y"]
            parts.append({
                "points": [[max(0, x - 2), max(0, y - 2)],
                           [min(39, x + 2), max(0, y - 2)],
                           [min(39, x + 2), min(39, y + 2)],
                           [max(0, x - 2), min(39, y + 2)]],
                "holes": [],
            })
        return {"parts": parts, "score": .8, "warnings": []}

    def test_fixed_family_covers_all_points_without_claiming_truth(self):
        stream = io.BytesIO()
        Image.new("RGB", (40, 40), "white").save(stream, "PNG")
        baseline, result, baseline_shape, shape, rows = evaluate_fixed_family_image(
            stream.getvalue(), ((0, 2), (1, 3)), self.response)
        self.assertGreater(baseline["area_pixels"], 0)
        self.assertTrue(baseline_shape["parts"])
        self.assertTrue(result["candidate_available"])
        self.assertEqual(result["positive_points_covered"], 4)
        self.assertEqual(result["negative_points_violated"], 0)
        self.assertEqual(result["additional_request_count"], 2)
        self.assertEqual(result["evaluation_request_count"], 3)
        self.assertEqual(result["baseline_retention"], 1.0)
        self.assertEqual(len(rows), 2)
        self.assertTrue(shape["parts"])

    def test_discovery_contract_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "unsupported contract"):
            derive_minimum_family({"schema_version": "wrong"})


if __name__ == "__main__":
    unittest.main()
