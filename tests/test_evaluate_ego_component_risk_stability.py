import io
import unittest

import numpy as np
from PIL import Image

from tools.evaluate_ego_component_risk_stability import gamma_090, jpeg_q95


class EgoComponentRiskStabilityTests(unittest.TestCase):
    def test_gamma_transform_is_bounded_and_shape_preserving(self):
        source = np.arange(256, dtype=np.float32).reshape(16, 16)
        transformed = gamma_090(source)
        self.assertEqual(transformed.shape, source.shape)
        self.assertGreaterEqual(float(transformed.min()), 0.0)
        self.assertLessEqual(float(transformed.max()), 255.0)

    def test_jpeg_transform_is_bounded_and_shape_preserving(self):
        source = np.arange(256, dtype=np.float32).reshape(16, 16)
        transformed = jpeg_q95(source)
        self.assertEqual(transformed.shape, source.shape)
        self.assertGreaterEqual(float(transformed.min()), 0.0)
        self.assertLessEqual(float(transformed.max()), 255.0)


if __name__ == "__main__":
    unittest.main()
