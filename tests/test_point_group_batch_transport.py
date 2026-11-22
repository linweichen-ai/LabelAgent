import io
import json
import unittest
from unittest import mock

from source.agent import workflow
from source.agent.workflow import SamHttpError, _sam_request, sam_point_groups


class PointGroupBatchTransportTest(unittest.TestCase):
    def test_single_point_fallback_uses_dedicated_point_service_url(self):
        response = io.BytesIO(json.dumps({"parts": []}).encode("utf-8"))
        response.__enter__ = lambda value: value
        response.__exit__ = lambda *_args: None
        config = {
            "sam3_url": "http://text-sam",
            "sam3_point_url": "http://point-sam",
        }
        with mock.patch(
                "source.agent.workflow.urllib.request.urlopen",
                return_value=response) as opened:
            _sam_request(
                config, "/segment_points", b"image",
                points=[{"x": 1, "y": 2, "label": 1}])
        self.assertEqual(
            opened.call_args.args[0].full_url,
            "http://point-sam/segment_points")

    def test_point_group_batch_uses_dedicated_point_service_url(self):
        response = io.BytesIO(json.dumps({
            "items": [],
            "diagnostics": {"image_encoded_once": True},
        }).encode("utf-8"))
        response.__enter__ = lambda value: value
        response.__exit__ = lambda *_args: None
        config = {
            "sam3_url": "http://text-sam",
            "sam3_point_url": "http://point-sam",
        }
        with mock.patch(
                "source.agent.workflow.urllib.request.urlopen",
                return_value=response) as opened:
            result = _sam_request(
                config, "/segment_point_groups_batch", b"image",
                prompt_groups=[[{"x": 1, "y": 2, "label": 1}]])
        self.assertTrue(result["diagnostics"]["image_encoded_once"])
        self.assertEqual(
            opened.call_args.args[0].full_url,
            "http://point-sam/segment_point_groups_batch")

    def test_group_client_accepts_complete_single_encoding_response(self):
        groups = [[{"x": 1, "y": 2, "label": 1}]]
        response = {"items": [{"index": 0, "points": groups[0],
                                "result": {"parts": []}}],
                    "diagnostics": {"image_encoded_once": True,
                                    "prompt_group_count": 1}}
        config = {"sam3_point_url": "http://batch-client"}
        with mock.patch.object(workflow, "sam_request",
                               return_value=response) as request:
            self.assertEqual(sam_point_groups(b"image", groups, config), response)
        request.assert_called_once_with(
            config, "/segment_point_groups_batch", b"image",
            prompt_groups=groups)

    def test_group_client_falls_back_only_for_missing_endpoint_and_caches_it(self):
        groups = [[{"x": 1, "y": 2, "label": 1}],
                  [{"x": 3, "y": 4, "label": 1}]]
        config = {"sam3_point_url": "http://legacy-point-client"}
        with workflow._sam_point_group_protocol_lock:
            workflow._sam_point_group_protocols.pop(
                config["sam3_point_url"], None)
        with mock.patch.object(workflow, "sam_request", side_effect=[
                SamHttpError(404, "Not Found"),
                {"parts": [{"points": [[1, 2]], "holes": []}]},
                SamHttpError(400, "SAM 3 没有找到目标，请补充提示点"),
                {"parts": [{"points": [[1, 2]], "holes": []}]},
                {"parts": [{"points": [[3, 4]], "holes": []}]},
        ]) as request:
            first = sam_point_groups(b"image", groups, config)
            second = sam_point_groups(b"image", groups, config)
        self.assertEqual(
            [call.args[1] for call in request.call_args_list],
            ["/segment_point_groups_batch", "/segment_points",
             "/segment_points", "/segment_points", "/segment_points"])
        self.assertFalse(first["diagnostics"]["image_encoded_once"])
        self.assertEqual(first["diagnostics"]["successful_count"], 1)
        self.assertIn("没有找到目标", first["items"][1]["error"])
        self.assertEqual(second["diagnostics"]["successful_count"], 2)

    def test_group_client_never_hides_busy_or_malformed_batch_failures(self):
        groups = [[{"x": 1, "y": 2, "label": 1}]]
        for suffix, failure in (
                ("busy", SamHttpError(503, "service busy")),
                ("malformed", {"items": [], "diagnostics": {}})):
            config = {"sam3_point_url": f"http://point-{suffix}"}
            with workflow._sam_point_group_protocol_lock:
                workflow._sam_point_group_protocols.pop(
                    config["sam3_point_url"], None)
            with self.subTest(suffix=suffix), mock.patch.object(
                    workflow, "sam_request",
                    side_effect=failure if isinstance(failure, Exception)
                    else None, return_value=None if isinstance(failure, Exception)
                    else failure) as request:
                with self.assertRaises((SamHttpError, ValueError)):
                    sam_point_groups(b"image", groups, config)
                request.assert_called_once()


if __name__ == "__main__":
    unittest.main()
