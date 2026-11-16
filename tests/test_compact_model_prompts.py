import io
import json
import unittest
from unittest.mock import patch
from PIL import Image
from source.agent import workflow


class CompactModelPromptTests(unittest.TestCase):
    def setUp(self):
        output = io.BytesIO()
        Image.new("RGB", (40, 40)).save(output, "JPEG")
        self.image = output.getvalue()
        self.shape = {"id": 1, "label": "pallet", "points": [[1, 1], [20, 1], [20, 20]]}
        self.description = {"description": "Pallet visible contours", "prompts": [{"label": "pallet", "text": "pallet"}]}
        self.reference = {"image": self.image, "marked_image": self.image, "annotations": [self.shape]}

    def test_compact_input_is_smaller_and_keeps_reference_and_target_evidence(self):
        calls = []
        for config in ({}, {"llm_prompt_profile": "compact"}):
            with patch.object(workflow, "chat", return_value={"status": "done", "reason": "No supported edit", "operations": []}) as call:
                workflow.review(config, self.image, self.image, self.description, [self.shape], ["pallet"], [], reference=self.reference)
                calls.append(call.call_args)
        self.assertLess(len(calls[1].args[1]), len(calls[0].args[1]) * 0.75)
        self.assertEqual(calls[1].kwargs["extra_images"][-2:], calls[0].kwargs["extra_images"][-2:])
        self.assertEqual(len(calls[1].kwargs["extra_images"]), 3)
        self.assertIn('"reference_counts":{"pallet":1}', calls[1].args[1])
        self.assertIn('"target_size":[40,40]', calls[1].args[1])

    def test_compact_profile_still_rejects_operations_on_unknown_candidates(self):
        before = json.dumps(self.shape)
        decision = {"status": "continue", "reason": "test", "operations": [{"operation": "remove", "shape_ids": [999], "reason": "test"}]}
        with patch.object(workflow, "chat", return_value=decision) as call:
            with self.assertRaises(ValueError):
                workflow.review({"llm_prompt_profile": "compact"}, self.image, self.image,
                                self.description, [self.shape], ["pallet"], [], reference=self.reference)
        self.assertEqual(call.call_count, workflow.MAX_PROTOCOL_RETRIES+1)
        self.assertEqual(json.dumps(self.shape), before)

    def test_sam_comparison_image_index_tracks_the_removed_empty_hole_panel(self):
        reply = {"status": "done", "reason": "No supported edit", "operations": [],
                 "sam_assessment": {"improved": False, "reason": "No verified improvement"}}
        with patch.object(workflow, "chat", return_value=reply) as call:
            workflow.review({"llm_prompt_profile": "compact"}, self.image, self.image,
                            self.description, [self.shape], ["pallet"], [], reference=self.reference,
                            sam_comparison={"shape_id": 1, "before_image": self.image})
        self.assertEqual(len(call.call_args.kwargs["extra_images"]), 4)
        self.assertIn('"sam_before_image":5', call.call_args.args[1])


if __name__ == "__main__":
    unittest.main()
