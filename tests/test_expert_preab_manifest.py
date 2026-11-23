"""Tests for candidate-only external expert pre-A/B queue freezing."""
import copy
import json
import unittest

from tools.build_expert_preab_manifest import build_preab_manifest


def source():
    return {
        "schema_version": "labelagent.slope-review-packet.v1",
        "mcap": "D:/private/factory.mcap",
        "mcap_sha256": "a" * 64,
        "topic": "/camera/image",
        "rows": [
            {"frame": 7, "cohort": "positive_proxy",
             "source_frame_sha256": "b" * 64,
             "raw_image": "raw/frame-7.jpg", "review_image": "review/7.jpg",
             "candidate_count": 1, "target_provenance": "machine_proxy",
             "review_status": "pending_human_review", "candidate_only": True,
             "human_truth": False, "approved_for_training": False,
             "training_return_allowed": False},
            {"frame": 2, "cohort": "negative_cohort",
             "source_frame_sha256": "c" * 64,
             "raw_image": "raw/frame-2.jpg", "review_image": "review/2.jpg",
             "candidate_count": 0, "target_provenance": "unverified_saved",
             "review_status": "pending_human_review", "candidate_only": True,
             "human_truth": False, "approved_for_training": False,
             "training_return_allowed": False},
        ],
    }


class ExpertPreAbManifestTests(unittest.TestCase):
    def test_freezes_deterministic_path_free_candidate_only_queue(self):
        encoded = json.dumps(source(), sort_keys=True).encode()
        first = build_preab_manifest(None, source_bytes=encoded)
        second = build_preab_manifest(None, source_bytes=encoded)
        self.assertEqual(first, second)
        self.assertEqual(first["sample_count"], 2)
        self.assertEqual(first["strata"], {"negative_cohort": 1,
                                            "positive_proxy": 1})
        self.assertEqual(first["pending_human_review"], 2)
        self.assertEqual(first["human_reference_count"], 0)
        self.assertFalse(first["ready_for_quality_ab"])
        self.assertFalse(first["external_upload_authorized"])
        self.assertFalse(first["images_copied"])
        serialized = json.dumps(first)
        self.assertNotIn("D:/private", serialized)
        self.assertNotIn("raw/frame", serialized)
        self.assertEqual([item["frame"] for item in first["samples"]], [2, 7])

    def test_rejects_test24_anywhere_in_source(self):
        item = source(); item["mcap"] = "D:/datasets/test24/data.mcap"
        with self.assertRaisesRegex(ValueError, "test24"):
            build_preab_manifest(None, source_bytes=json.dumps(item).encode())

    def test_rejects_human_truth_training_and_upload_boundary_drift(self):
        for field, value in (("candidate_only", False), ("human_truth", True),
                             ("approved_for_training", True),
                             ("training_return_allowed", True)):
            item = source(); item["rows"][0][field] = value
            with self.assertRaisesRegex(ValueError, "candidate-only"):
                build_preab_manifest(None, source_bytes=json.dumps(item).encode())

    def test_rejects_duplicates_and_non_pending_rows(self):
        duplicate = source(); duplicate["rows"][1]["frame"] = 7
        with self.assertRaisesRegex(ValueError, "identity"):
            build_preab_manifest(None, source_bytes=json.dumps(duplicate).encode())
        completed = source(); completed["rows"][0]["review_status"] = "accepted"
        with self.assertRaisesRegex(ValueError, "pending"):
            build_preab_manifest(None, source_bytes=json.dumps(completed).encode())

    def test_input_is_not_mutated(self):
        item = source(); before = copy.deepcopy(item)
        build_preab_manifest(None, source_bytes=json.dumps(item).encode())
        self.assertEqual(item, before)


if __name__ == "__main__":
    unittest.main()
