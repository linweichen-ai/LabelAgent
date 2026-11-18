import json
import unittest

from tools.build_cvat_other_robot_distillation import (
    frame_split, generation_contract, json_sha256, target_for_count)


class CvatOtherRobotDistillationTests(unittest.TestCase):
    def test_generation_contract_hash_and_registry(self):
        contract = generation_contract()
        body = {key: value for key, value in contract.items()
                if key != "contract_sha256"}
        self.assertEqual(contract["contract_sha256"], json_sha256(body))
        self.assertEqual(contract["labels"], ["other_robot"])
        self.assertEqual(
            contract["response_schema"]["properties"]["assessments"]["required"],
            ["other_robot"])

    def test_frame_split_has_disjoint_contiguous_blocks_and_gaps(self):
        self.assertEqual(frame_split(0), "train")
        self.assertIsNone(frame_split(18))
        self.assertEqual(frame_split(19), "dev")
        self.assertEqual(frame_split(31), "dev")
        self.assertIsNone(frame_split(32))
        self.assertIsNone(frame_split(47))
        self.assertEqual(frame_split(48), "holdout")
        self.assertEqual(frame_split(66), "holdout")
        self.assertIsNone(frame_split(67))
        self.assertIsNone(frame_split(100))
        self.assertEqual(frame_split(101), "dev")
        self.assertEqual(frame_split(109), "dev")
        self.assertEqual(frame_split(126), "holdout")
        self.assertEqual(frame_split(134), "holdout")
        self.assertIsNone(frame_split(135))
        self.assertEqual(frame_split(149), "train")

    def test_target_marks_presence_without_promoting_human_truth(self):
        present = json.loads(target_for_count(2))
        absent = json.loads(target_for_count(0))
        self.assertEqual(
            present["assessments"]["other_robot"]["evidence"], "present")
        self.assertEqual(
            absent["assessments"]["other_robot"]["evidence"], "absent")


if __name__ == "__main__":
    unittest.main()
