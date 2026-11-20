import io
import unittest

from PIL import Image

from tools import benchmark_visible_vehicle_semantic_fast_route as benchmark


class VisibleVehicleSemanticFastRouteTests(unittest.TestCase):
    def test_compact_overview_preserves_aspect_and_fast_route_bound(self):
        source = io.BytesIO()
        Image.new("RGB", (1920, 1536), "gray").save(source, "PNG")
        value = benchmark.compact_overview(source.getvalue())
        with Image.open(io.BytesIO(value)) as result:
            self.assertEqual(result.size, (640, 512))

    def test_compact_overview_supports_larger_experiment_bound(self):
        source = io.BytesIO()
        Image.new("RGB", (1920, 1536), "white").save(source, "JPEG")
        value = benchmark.compact_overview(source.getvalue(), (960, 768))
        with Image.open(io.BytesIO(value)) as result:
            self.assertEqual(result.size, (960, 768))

    def test_semantic_route_fails_closed(self):
        labels = ["车辆", "other_robot"]
        self.assertEqual(benchmark.semantic_route({
            "classification": "conventional_vehicle", "confidence": .95,
        }, labels), "车辆")
        self.assertEqual(benchmark.semantic_route({
            "classification": "partial_fragment", "confidence": .99,
        }, labels), "rejected")
        self.assertEqual(benchmark.semantic_route({
            "classification": "autonomous_robot_or_agv", "confidence": .89,
        }, labels), "rejected")


if __name__ == "__main__":
    unittest.main()
