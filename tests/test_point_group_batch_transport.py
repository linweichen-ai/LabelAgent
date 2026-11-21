import io
import json
import unittest
from unittest import mock

from source.agent.workflow import _sam_request


class PointGroupBatchTransportTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
