import unittest

import numpy as np

from tools.rank_ego_component_review_risk import (
    _crop_bounds, _percentile_ranks, grayscale_features, rank_rows)


class EgoComponentReviewRiskTests(unittest.TestCase):
    def test_crop_bounds_are_clamped_at_image_edge(self):
        self.assertEqual(_crop_bounds([0, 0, 1, 1], 40, 30), (0, 0, 32, 30))

    def test_grayscale_features_are_finite(self):
        gray = np.arange(64 * 64, dtype=np.float32).reshape(64, 64) % 255
        component = np.zeros_like(gray, dtype=bool)
        component[30, 30] = True
        original = np.ones_like(component)
        original[30, 30] = False
        features = grayscale_features(
            gray, component, [30, 30, 31, 31], original)
        for key, value in features.items():
            if key != "context_bbox_xyxy":
                self.assertTrue(np.isfinite(value))

    def test_percentile_ranks_average_ties(self):
        self.assertEqual(_percentile_ranks([1.0, 1.0, 3.0]),
                         [0.25, 0.25, 1.0])

    def test_ranking_never_makes_an_automatic_decision(self):
        rows = [
            {"component_id": "dark", "camera": "camera1",
             "area_pixels": 1, "bbox_elongation": 1.0,
             "target_luma_mean": 5.0, "context_luma_std": 2.0,
             "context_laplacian_variance": 1.0,
             "context_gradient_mean": 1.0,
             "context_edge_coherence": 0.1,
             "gap_distance_pixels": 1.0,
             "context_background_density": 0.5,
             "weak_external_support_packages": 1},
            {"component_id": "clear", "camera": "camera1",
             "area_pixels": 10, "bbox_elongation": 1.0,
             "target_luma_mean": 150.0, "context_luma_std": 30.0,
             "context_laplacian_variance": 100.0,
             "context_gradient_mean": 20.0,
             "context_edge_coherence": 0.9,
             "gap_distance_pixels": 5.0,
             "context_background_density": 0.1,
             "weak_external_support_packages": 0},
        ]
        ranked = rank_rows(rows)
        self.assertEqual(ranked[0]["component_id"], "dark")
        self.assertEqual(ranked[0]["camera_review_priority_rank"], 1)
        self.assertTrue(all(row["automatic_decision"] is None for row in ranked))


if __name__ == "__main__":
    unittest.main()
