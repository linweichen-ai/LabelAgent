import unittest

from tools.evaluate_traversable_union_static import summarize


class TraversableUnionStaticTests(unittest.TestCase):
    def test_summary_keeps_floor_and_slope_coverage_separate(self):
        rows = [
            {"assistant_ramp_point": {"x": 1, "y": 1},
             "assistant_ramp_point_covered": True,
             "assistant_ramp_point_covered_by_floor": True,
             "assistant_ramp_point_covered_by_slope": False,
             "floor_candidate_count": 1, "slope_candidate_count": 0,
             "elapsed_seconds": 1.0},
            {"assistant_ramp_point": {"x": 2, "y": 2},
             "assistant_ramp_point_covered": True,
             "assistant_ramp_point_covered_by_floor": False,
             "assistant_ramp_point_covered_by_slope": True,
             "floor_candidate_count": 1, "slope_candidate_count": 1,
             "elapsed_seconds": 1.0},
            {"assistant_ramp_point": None,
             "assistant_ramp_point_covered": None,
             "assistant_ramp_point_covered_by_floor": None,
             "assistant_ramp_point_covered_by_slope": None,
             "floor_candidate_count": 1, "slope_candidate_count": 0,
             "elapsed_seconds": 1.0},
        ]
        result = summarize(rows)
        self.assertEqual(result["ramp_points_covered_by_traversable_union"], 2)
        self.assertEqual(result["ramp_points_covered_by_floor"], 1)
        self.assertEqual(result["ramp_points_covered_by_slope"], 1)


if __name__ == "__main__":
    unittest.main()
