import io
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from source.sam3_service import point_segment
from source.sam3_service import server


def image_bytes(width=32, height=24):
    output = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(output, "PNG")
    return output.getvalue()


class FakeProcessor:
    def __init__(self):
        self.calls = 0

    def set_image(self, image):
        self.calls += 1
        return {"size": image.size}


class FakeModel:
    def __init__(self):
        self.calls = []

    def predict_inst(self, state, point_coords, point_labels, multimask_output):
        self.calls.append((point_coords.copy(), point_labels.copy(), multimask_output))
        width, height = state["size"]
        x, y = [int(value) for value in point_coords[0]]
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[max(0, y - 3):min(height, y + 4),
             max(0, x - 3):min(width, x + 4)] = 1
        return np.asarray([mask]), np.asarray([0.9]), None


class PointSegmentBatchTests(unittest.TestCase):
    def test_batch_reuses_one_image_encoding_and_keeps_candidates_independent(self):
        model, processor = FakeModel(), FakeProcessor()
        with mock.patch.object(point_segment, "_predictor", return_value=(model, processor)):
            result = point_segment.segment_many(
                image_bytes(), [{"x": 8, "y": 8}, {"x": 24, "y": 16}])

        self.assertEqual(processor.calls, 1)
        self.assertEqual(len(model.calls), 2)
        self.assertTrue(result["diagnostics"]["image_encoded_once"])
        self.assertEqual(result["diagnostics"]["point_count"], 2)
        self.assertEqual(result["diagnostics"]["successful_count"], 2)
        self.assertEqual([item["point"] for item in result["items"]],
                         [{"x": 8, "y": 8}, {"x": 24, "y": 16}])
        for item in result["items"]:
            self.assertIn("result", item)
            self.assertEqual(item["result"]["warnings"], [])
            self.assertEqual(len(item["result"]["parts"]), 1)

    def test_batch_rejects_duplicate_and_out_of_bounds_points(self):
        with self.assertRaisesRegex(ValueError, "不得重复"):
            point_segment.segment_many(
                image_bytes(), [{"x": 8, "y": 8}, {"x": 8, "y": 8}])
        with self.assertRaisesRegex(ValueError, "超出图像范围"):
            point_segment.segment_many(image_bytes(), [{"x": 32, "y": 8}])

    def test_http_batch_endpoint_preserves_the_bounded_contract(self):
        payload = {"items": [], "diagnostics": {
            "image_encoded_once": True, "point_count": 1,
            "successful_count": 0, "total_seconds": 0.1}}
        import base64

        with mock.patch.object(server, "segment_many", return_value=payload) as run:
            response = server.app.test_client().post("/segment_points_batch", json={
                "image": base64.b64encode(image_bytes()).decode("ascii"),
                "points": [{"x": 8, "y": 8}],
            })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), payload)
        self.assertEqual(run.call_args.args[1], [{"x": 8, "y": 8}])

if __name__ == "__main__":
    unittest.main()
