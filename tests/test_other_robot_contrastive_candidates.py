import hashlib
from pathlib import Path
import tempfile
import unittest

from tools.build_other_robot_contrastive_candidates import build_rows, validate_spec


class OtherRobotContrastiveCandidateTests(unittest.TestCase):
    def test_balanced_candidate_rows_stay_non_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            samples = []
            for evidence in ("present", "absent"):
                image = Path(directory) / f"{evidence}.png"
                image.write_bytes(evidence.encode())
                samples.append({
                    "sample_id": evidence, "local_image_path": str(image),
                    "remote_image_path": f"/remote/{evidence}.png",
                    "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                    "training_reason": f"visible evidence is {evidence}",
                    "assistant_review_candidate": {
                        "review_kind": "assistant_visual_candidate",
                        "human_truth": False, "expected_evidence": evidence,
                        "reason": f"candidate says {evidence}",
                    },
                })
            spec = {
                "schema_version": "labelagent.other-robot-contrastive-spec.v1",
                "human_truth": False, "candidate_only": True,
                "production_training_approved": False, "samples": samples,
            }
            rows = build_rows(validate_spec(spec), 2, "inspect")
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(row["human_truth"] is False for row in rows))
            self.assertEqual({row["candidate_kind"] for row in rows},
                             {"assistant_review_candidate"})

    def test_one_sided_spec_is_rejected(self):
        spec = {
            "schema_version": "labelagent.other-robot-contrastive-spec.v1",
            "human_truth": False, "candidate_only": True,
            "production_training_approved": False,
            "samples": [{
                "sample_id": "positive", "local_image_path": "image.png",
                "remote_image_path": "/remote/image.png", "image_sha256": "0" * 64,
                "training_reason": "a separate robot is visible",
                "assistant_review_candidate": {
                    "review_kind": "assistant_visual_candidate",
                    "human_truth": False, "expected_evidence": "present",
                    "reason": "a separate robot is visible",
                },
            }],
        }
        with self.assertRaises(ValueError):
            validate_spec(spec)


if __name__ == "__main__":
    unittest.main()
