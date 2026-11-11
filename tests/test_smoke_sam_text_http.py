import unittest

from tools.smoke_sam_text_http import validate_results


class SamTextSmokeTests(unittest.TestCase):
    def test_accepts_nonempty_in_bounds_parts(self):
        payload = {"results": [{"score": 0.9, "parts": [{
            "points": [[0, 0], [9, 0], [9, 7]],
            "holes": [[[2, 2], [3, 2], [3, 3]]],
        }]}]}
        summary = validate_results(payload, 10, 8)
        self.assertEqual(summary["result_count"], 1)
        self.assertEqual(summary["part_count"], 1)
        self.assertEqual(summary["outer_vertex_count"], 3)
        self.assertEqual(summary["hole_count"], 1)

    def test_rejects_http_success_with_no_masks(self):
        with self.assertRaisesRegex(ValueError, "no results"):
            validate_results({"results": []}, 1920, 1536)

    def test_rejects_out_of_bounds_vertices(self):
        payload = {"results": [{"score": 0.8, "parts": [{
            "points": [[0, 0], [1920, 10], [10, 10]], "holes": [],
        }]}]}
        with self.assertRaisesRegex(ValueError, "outside"):
            validate_results(payload, 1920, 1536)


if __name__ == "__main__":
    unittest.main()
