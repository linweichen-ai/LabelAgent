import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.augment_other_robot_hard_negatives import augment_rows, validate_spec


class OtherRobotHardNegativeTests(unittest.TestCase):
    def test_candidate_negative_is_repeated_without_truth_promotion(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "image.png"
            image.write_bytes(b"not-a-real-image-needed-for-boundary-test")
            spec = {
                "schema_version": "labelagent.other-robot-hard-negative-spec.v1",
                "human_truth": False, "candidate_only": True,
                "production_training_approved": False,
                "samples": [{
                    "sample_id": "negative-1",
                    "local_image_path": str(image),
                    "remote_image_path": "/remote/image.png",
                    "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                    "training_reason": "ego structure does not support a separate robot",
                    "assistant_review_candidate": {
                        "review_kind": "assistant_visual_candidate",
                        "human_truth": False, "expected_evidence": "absent",
                        "reason": "Only the ego robot is visible.",
                    },
                }],
            }
            samples = validate_spec(spec)
            rows = augment_rows([], samples, 3, "inspect")
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(row["candidate_kind"] ==
                                "assistant_review_candidate" for row in rows))
            self.assertTrue(all(row["human_truth"] is False for row in rows))
            self.assertTrue(all(row["production_training_approved"] is False
                                for row in rows))
            self.assertEqual({json.loads(row["target"])["assessments"]
                              ["other_robot"]["evidence"] for row in rows},
                             {"absent"})
            self.assertEqual({json.loads(row["target"])["assessments"]
                              ["other_robot"]["reason"] for row in rows},
                             {"ego structure does not support a separate robot"})

    def test_promoted_review_is_rejected(self):
        spec = {
            "schema_version": "labelagent.other-robot-hard-negative-spec.v1",
            "human_truth": False, "candidate_only": True,
            "production_training_approved": False,
            "samples": [{
                "sample_id": "negative-1", "local_image_path": "image.png",
                "remote_image_path": "/remote/image.png", "image_sha256": "0" * 64,
                "training_reason": "ego structure does not support a separate robot",
                "assistant_review_candidate": {
                    "review_kind": "assistant_visual_candidate",
                    "human_truth": True, "expected_evidence": "absent",
                    "reason": "Only the ego robot is visible.",
                },
            }],
        }
        with self.assertRaises(ValueError):
            validate_spec(spec)


if __name__ == "__main__":
    unittest.main()
