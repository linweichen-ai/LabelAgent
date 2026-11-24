import json
import tempfile
import unittest
from pathlib import Path

from tools import build_new_class_candidate_pool as pool


class NewClassCandidatePoolTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.policy = self.root / "policy.json"
        self.policy.write_text(json.dumps({
            "schema_version": pool.CONFIG_SCHEMA,
            "known_labels": {
                "pallet": ["栈板"],
                "person": ["人", "pedestrian"],
                "other_robot": ["other robot", "forklift"],
                "floor": ["ground"],
                "slope": ["ramp"],
            },
            "candidate_synonym_groups": {
                "rack": ["shelf", "shelving", "货架"]
            },
            "minimum_distinct_frames": 2,
            "minimum_distinct_sequences": 1,
            "automatic_registry_mutation": False,
        }), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def observations(self, rows):
        path = self.root / "observations.json"
        path.write_text(json.dumps({
            "schema_version": pool.INPUT_SCHEMA,
            "observations": rows,
            "candidate_only": True,
            "human_truth": False,
            "approved_for_training": False,
            "test24_used": False,
        }), encoding="utf-8")
        return path

    @staticmethod
    def row(number, names, *, sequence="003", split="dev", image_hash=None):
        return {
            "observation_id": f"obs-{number}",
            "sample_id": f"sample-{number}",
            "sequence": sequence,
            "camera": number % 4,
            "frame_id": str(100 + number),
            "split": split,
            "image_sha256": image_hash or f"{number + 1:064x}",
            "unknown_candidates": [
                {"candidate_name": name, "evidence": "present",
                 "confidence": 0.7 + number / 100, "reason": "visible structure"}
                for name in names
            ],
        }

    def test_aggregates_explicit_synonyms_across_frames(self):
        source = self.observations([
            self.row(0, ["rack", "shelf"]),
            self.row(1, ["货架"], sequence="004"),
            self.row(2, ["chair"]),
        ])
        result = pool.build_pool(source, self.policy)
        rack = next(item for item in result["candidates"]
                    if item["canonical_candidate_name"] == "rack")
        chair = next(item for item in result["candidates"]
                     if item["canonical_candidate_name"] == "chair")
        self.assertEqual(rack["distinct_frames"], 2)
        self.assertEqual(rack["distinct_sequences"], 2)
        self.assertEqual(rack["status"], "human_downstream_value_review_required")
        self.assertEqual(chair["status"], "insufficient_cross_frame_support")
        self.assertEqual(result["summary"]["human_review_queue"], 1)
        self.assertFalse(result["registry_mutated"])

    def test_known_label_aliases_never_become_new_classes(self):
        source = self.observations([
            self.row(0, ["forklift", "other robot"]),
            self.row(1, ["other_robot"]),
        ])
        result = pool.build_pool(source, self.policy)
        self.assertEqual(result["candidates"], [])
        self.assertEqual(len(result["known_label_conflicts"]), 1)
        self.assertEqual(result["known_label_conflicts"][0]
                         ["canonical_candidate_name"], "other_robot")
        self.assertEqual(result["summary"]["automatic_registry_mutations"], 0)

    def test_deduplicates_same_frame_aliases_and_ignores_absent(self):
        first = self.row(0, ["rack", "shelf"])
        first["unknown_candidates"].append(
            {"candidate_name": "chair", "evidence": "absent", "confidence": 0.9})
        result = pool.build_pool(self.observations([first]), self.policy)
        self.assertEqual(result["candidates"][0]["distinct_frames"], 1)
        self.assertEqual(result["summary"]["ignored_absent_mentions"], 1)

    def test_rejects_test24_and_unsafe_boundaries(self):
        with self.assertRaisesRegex(ValueError, "test24 observation"):
            pool.build_pool(self.observations([self.row(0, ["rack"], split="test24")]),
                            self.policy)
        source = self.observations([self.row(1, ["rack"])])
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload["approved_for_training"] = True
        source.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unsafe observation manifest boundary"):
            pool.build_pool(source, self.policy)

    def test_rejects_ambiguous_alias_configuration(self):
        policy_payload = json.loads(self.policy.read_text(encoding="utf-8"))
        policy_payload["candidate_synonym_groups"]["storage"] = ["shelf"]
        self.policy.write_text(json.dumps(policy_payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "ambiguous alias"):
            pool.build_pool(self.observations([self.row(0, ["rack"])]), self.policy)

    def test_writes_review_template_without_registry_authority(self):
        source = self.observations([self.row(0, ["rack"]), self.row(1, ["shelf"])])
        output = self.root / "output"
        result = pool.write_outputs(source, self.policy, output)
        template = json.loads((output / "human-ontology-review-template.json")
                              .read_text(encoding="utf-8"))
        self.assertEqual(result["summary"]["human_review_queue"], 1)
        self.assertEqual(len(template["rows"]), 1)
        self.assertFalse(template["automatic_registry_mutation"])
        self.assertFalse(template["registry_mutated"])
        self.assertFalse(template["approved_for_training"])
        self.assertTrue((output / "receipt.json").is_file())

    def test_valid_review_only_creates_manual_registry_proposal(self):
        source = self.observations([self.row(0, ["rack"]), self.row(1, ["shelf"])])
        output = self.root / "output"
        pool.write_outputs(source, self.policy, output)
        template_path = output / "human-ontology-review-template.json"
        submission = json.loads(template_path.read_text(encoding="utf-8"))
        submission["reviewer_declared_id"] = "reviewer-1"
        submission["rows"][0].update({"ontology_decision": "accept_candidate",
                                      "downstream_value": "yes"})
        submission_path = self.root / "submission.json"
        submission_path.write_text(json.dumps(submission), encoding="utf-8")
        result = pool.validate_review_submission(output / "candidate-pool.json",
                                                 submission_path)
        self.assertEqual(result["manual_registry_proposals"], ["rack"])
        self.assertTrue(result["manual_registry_change_required"])
        self.assertEqual(result["automatic_registry_mutations"], 0)
        self.assertFalse(result["registry_mutated"])

    def test_review_validation_rejects_hash_drift_and_automatic_mutation(self):
        source = self.observations([self.row(0, ["rack"]), self.row(1, ["shelf"])])
        output = self.root / "output"
        pool.write_outputs(source, self.policy, output)
        template_path = output / "human-ontology-review-template.json"
        submission = json.loads(template_path.read_text(encoding="utf-8"))
        submission["reviewer_declared_id"] = "reviewer-1"
        submission["rows"][0].update({"ontology_decision": "reject_candidate",
                                      "downstream_value": "no"})
        submission["candidate_pool_sha256"] = "0" * 64
        submission_path = self.root / "submission.json"
        submission_path.write_text(json.dumps(submission), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "bound to another"):
            pool.validate_review_submission(output / "candidate-pool.json",
                                            submission_path)
        submission["candidate_pool_sha256"] = pool.file_sha256(
            output / "candidate-pool.json")
        submission["registry_mutated"] = True
        submission_path.write_text(json.dumps(submission), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unsafe ontology review boundary"):
            pool.validate_review_submission(output / "candidate-pool.json",
                                            submission_path)


if __name__ == "__main__":
    unittest.main()
