import unittest

from tools.split_sam3_assistant_crop_dataset import split


class AssistantCropSplitTests(unittest.TestCase):
    def test_whole_earliest_group_is_held_out(self):
        indices = list(range(15)) + list(range(37, 93))
        images = [{"id": number + 1, "candidate_id": f"{index}:x",
                   "file_name": f"images/{number + 1}.png"}
                  for number, index in enumerate(indices)]
        coco = {"images": images, "annotations": [
            {"id": row["id"], "image_id": row["id"], "category_id": 1}
            for row in images], "categories": [{"id": 1, "name": "pallet"}]}
        manifest = {"schema_version": "labelagent.sam3-assistant-instance-crop-dataset.v1",
                    "crop_count": 71, "human_truth": False,
                    "test24_included": False}
        train, holdout, receipt = split(coco, manifest)
        self.assertEqual(len(train["images"]), 56)
        self.assertEqual(len(holdout["images"]), 15)
        self.assertEqual(receipt["holdout_candidate_ids"][0], "0:x")
        self.assertEqual(receipt["holdout_candidate_ids"][-1], "14:x")
        self.assertFalse(receipt["human_truth"])


if __name__ == "__main__":
    unittest.main()
