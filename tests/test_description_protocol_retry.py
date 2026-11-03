import io
from pathlib import Path
import unittest
from unittest.mock import patch
from PIL import Image
from source.agent import workflow, server


class DescriptionProtocolRetryTests(unittest.TestCase):
    def setUp(self):
        output = io.BytesIO()
        Image.new("RGB", (40, 40)).save(output, "JPEG")
        self.image = output.getvalue()
        self.shapes = [{"id": 1, "label": "pallet", "points": [[1, 1], [20, 1], [20, 20]]}]

    def test_missing_class_prompt_is_sent_back_for_correction(self):
        valid = {"description": "Visible pallet outlines", "prompts": [{"label": "pallet", "text": "pallet"}]}
        with patch.object(workflow, "chat", side_effect=[{"description": "objects", "prompts": []}, valid]) as call:
            result = workflow.describe({"llm_prompt_profile": "compact"}, self.image, self.shapes, ["pallet"], self.image)
        self.assertEqual(result, valid)
        self.assertEqual(call.call_count, 2)
        self.assertIn("pallet", call.call_args_list[1].args[1])

    def test_transport_failure_is_not_retried_as_bad_model_format(self):
        with patch.object(workflow, "chat", side_effect=ValueError("connection failed")) as call:
            with self.assertRaisesRegex(ValueError, "connection failed"):
                workflow.describe({}, self.image, self.shapes, ["pallet"], self.image)
        self.assertEqual(call.call_count, 1)

    def test_prompt_profile_change_invalidates_cached_reference_description(self):
        class Media:
            path = Path("example.mcap")
        state = {"video": Media(), "topic": "image", "truth": {"0": self.shapes}, "config": {}}
        with patch.dict(server.state, state, clear=True):
            full = server.reference_signature(0)
            server.state["config"]["llm_prompt_profile"] = "compact"
            self.assertNotEqual(full, server.reference_signature(0))
            server.state["config"]["llm_prompt_profile"] = "full"
            self.assertEqual(full, server.reference_signature(0))


if __name__ == "__main__":
    unittest.main()
