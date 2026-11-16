import io
import json
import math
import unittest
from unittest.mock import patch
from PIL import Image
from source.agent import workflow


class ReferencePromptBudgetTests(unittest.TestCase):
    def test_dense_geometry_is_summarized_without_mutating_reference_or_dropping_outline(self):
        points = [[20+12*math.cos(i*math.tau/2000), 20+12*math.sin(i*math.tau/2000)] for i in range(2000)]
        annotations = [{"id": 1, "label": "pallet", "points": points}]
        before = json.dumps(annotations)
        output = io.BytesIO()
        Image.new("RGB", (40, 40)).save(output, "JPEG")
        image = output.getvalue()
        reply = {"description": "Pallet contours", "prompts": [{"label": "pallet", "text": "pallet"}]}
        with patch.object(workflow, "chat", return_value=reply) as call:
            workflow.describe({}, image, annotations, ["pallet"], marked_image=image)
        self.assertLess(len(call.call_args.args[1]), 10000)
        self.assertIn('"vertices": 2000', call.call_args.args[1])
        self.assertEqual(call.call_args.kwargs["extra_images"], (image,))
        self.assertEqual(json.dumps(annotations), before)


if __name__ == "__main__":
    unittest.main()
