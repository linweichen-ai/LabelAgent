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

    def predict_inst(self, state, point_coords, point_labels, multimask_output,
                     box=None):
        coords_copy = point_coords.copy() if point_coords is not None else None
        labels_copy = point_labels.copy() if point_labels is not None else None
        box_copy = box.copy() if box is not None else None
        self.calls.append((coords_copy, labels_copy, multimask_output, box_copy))
        width, height = state["size"]
        if point_coords is not None:
            x, y = [int(value) for value in point_coords[0]]
        else:
            x = int((box[0] + box[2]) / 2)
            y = int((box[1] + box[3]) / 2)
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

    def test_box_only_prompt_reaches_predictor_and_can_be_refined_with_points(self):
        model, processor = FakeModel(), FakeProcessor()
        with mock.patch.object(
                point_segment, "_predictor", return_value=(model, processor)):
            box_only = point_segment.segment(
                image_bytes(), [], box=[4, 5, 20, 18])
            refined = point_segment.segment(
                image_bytes(), [{"x": 8, "y": 8, "label": 1}],
                box=[4, 5, 20, 18])

        self.assertEqual(box_only["prompt_box"], [4.0, 5.0, 20.0, 18.0])
        self.assertEqual(refined["prompt_box"], [4.0, 5.0, 20.0, 18.0])
        self.assertIsNone(model.calls[0][0])
        self.assertEqual(model.calls[0][3].tolist(), [4, 5, 20, 18])
        self.assertEqual(model.calls[1][0].tolist(), [[8, 8]])
        self.assertEqual(processor.calls, 2)

    def test_box_prompt_fails_closed_before_model_work(self):
        invalid = ([1, 2, 3], [10, 5, 4, 18], [-1, 2, 10, 12],
                   [1, 2, float("inf"), 12], [1, 2, 2, 12])
        for box in invalid:
            with self.subTest(box=box), mock.patch.object(
                    point_segment, "_predictor") as predictor:
                with self.assertRaises(ValueError):
                    point_segment.segment(image_bytes(), [], box=box)
                predictor.assert_not_called()

    def test_http_point_endpoint_forwards_optional_box(self):
        payload = {"points": [[4, 5], [20, 5], [20, 18]],
                   "holes": [], "parts": [], "score": .9,
                   "warnings": [], "prompt_box": [4, 5, 20, 18]}
        import base64

        with mock.patch.object(server, "segment", return_value=payload) as run:
            response = server.app.test_client().post("/segment_points", json={
                "image": base64.b64encode(image_bytes()).decode("ascii"),
                "points": [], "box": [4, 5, 20, 18],
            })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), payload)
        self.assertEqual(run.call_args.kwargs["box"], [4, 5, 20, 18])

    def test_prompt_groups_reuse_one_encoding_and_preserve_negative_points(self):
        model, processor = FakeModel(), FakeProcessor()
        groups = [
            [{"x": 8, "y": 8, "label": 1},
             {"x": 24, "y": 20, "label": 0}],
            [{"x": 24, "y": 8, "label": 1},
             {"x": 8, "y": 20, "label": 0}],
        ]
        with mock.patch.object(
                point_segment, "_predictor", return_value=(model, processor)):
            result = point_segment.segment_prompt_groups(image_bytes(), groups)

        self.assertEqual(processor.calls, 1)
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(model.calls[0][1].tolist(), [1, 0])
        self.assertFalse(model.calls[0][2])
        self.assertTrue(result["diagnostics"]["image_encoded_once"])
        self.assertEqual(result["diagnostics"]["prompt_group_count"], 2)
        self.assertEqual(result["diagnostics"]["successful_count"], 2)
        self.assertEqual([item["points"] for item in result["items"]], groups)
        self.assertEqual([item["index"] for item in result["items"]], [0, 1])

    def test_prompt_group_results_match_the_existing_sequential_contract(self):
        groups = [
            [{"x": 8, "y": 8, "label": 1},
             {"x": 24, "y": 20, "label": 0}],
            [{"x": 24, "y": 8, "label": 1},
             {"x": 8, "y": 20, "label": 0}],
        ]
        batch_model, batch_processor = FakeModel(), FakeProcessor()
        with mock.patch.object(
                point_segment, "_predictor",
                return_value=(batch_model, batch_processor)):
            batch = point_segment.segment_prompt_groups(image_bytes(), groups)

        sequential_model, sequential_processor = FakeModel(), FakeProcessor()
        with mock.patch.object(
                point_segment, "_predictor",
                return_value=(sequential_model, sequential_processor)):
            sequential = [
                point_segment.segment(image_bytes(), group) for group in groups
            ]

        self.assertEqual(
            [item["result"] for item in batch["items"]], sequential)
        self.assertEqual(batch_processor.calls, 1)
        self.assertEqual(sequential_processor.calls, len(groups))

    def test_prompt_groups_fail_closed_before_model_work(self):
        invalid = [
            [[{"x": 8, "y": 8, "label": 0}]],
            [[{"x": 8, "y": 8, "label": 2}]],
            [[{"x": 32, "y": 8, "label": 1}]],
            [[]],
        ]
        for groups in invalid:
            with self.subTest(groups=groups), mock.patch.object(
                    point_segment, "_predictor") as predictor:
                with self.assertRaises(ValueError):
                    point_segment.segment_prompt_groups(image_bytes(), groups)
                predictor.assert_not_called()

    def test_http_prompt_group_endpoint_keeps_groups_bounded(self):
        payload = {"items": [], "diagnostics": {
            "image_encoded_once": True, "prompt_group_count": 1,
            "successful_count": 0, "total_seconds": 0.1}}
        import base64

        groups = [[{"x": 8, "y": 8, "label": 1},
                   {"x": 24, "y": 16, "label": 0}]]
        with mock.patch.object(
                server, "segment_prompt_groups", return_value=payload) as run:
            response = server.app.test_client().post(
                "/segment_point_groups_batch", json={
                    "image": base64.b64encode(image_bytes()).decode("ascii"),
                    "prompt_groups": groups,
                })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), payload)
        self.assertEqual(run.call_args.args[1], groups)

    def test_remote_http_prompt_group_endpoint_uses_the_same_contract(self):
        # The local workbench environment intentionally does not install the
        # remote-only FastAPI stack.  Keep this dependency-free guard so the
        # remote deployment entrypoint cannot silently omit the new route.
        remote_source = (
            point_segment.Path(__file__).resolve().parents[1]
            / "source" / "sam3_service" / "remote_fastapi_server.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            '@app.post("/segment_point_groups_batch")', remote_source)
        self.assertIn(
            "with request_lock:\n            return segment_prompt_groups(",
            remote_source)

if __name__ == "__main__":
    unittest.main()
