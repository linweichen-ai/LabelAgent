import hashlib
import json
import unittest

from tools.collect_vlm_teacher_candidates import (
    canonical_teacher_target,
    frozen_train_images,
    teacher_candidate_record,
)


def manifest():
    images = []
    identifier = 1
    for split, sequences in (("train", range(9)), ("test", range(10, 13))):
        for sequence in sequences:
            for index in range(8):
                payload = f"{split}-{sequence}-{index}".encode()
                images.append({
                    "image_id": identifier,
                    "split": split,
                    "sequence": f"{sequence:03d}",
                    "frame_id": str(index // 4),
                    "camera": f"camera_{index % 4}",
                    "file_name": f"{sequence:03d}_{index}.png",
                    "source_path": f"/safe/{sequence:03d}_{index}.png",
                    "source_sha256": hashlib.sha256(payload).hexdigest(),
                })
                identifier += 1
    return {"preannotations_are_ground_truth": False, "images": images}


def discovery(flags=None):
    return {
        "assessments": [
            {"label": "pallet", "evidence": "present", "confidence": 0.9, "reason": "pallet edge visible"},
            {"label": "person", "evidence": "absent", "confidence": 0.8, "reason": "no person pixels visible"},
        ],
        "unknown_candidates": [{
            "candidate_name": "barrel", "evidence": "blue cylinder visible",
            "confidence": 0.7, "route": "human_review_required",
            "ontology_mutation_allowed": False,
        }],
        "consistency_flags": flags or [],
        "registry_conflicts": [],
        "open_set_scan_complete": True,
    }


class VlmTeacherCandidateTests(unittest.TestCase):
    def test_round_robin_pilot_uses_train_only_and_covers_sequences(self):
        selected = frozen_train_images(manifest(), 9)
        self.assertEqual(len(selected), 9)
        self.assertEqual({row["sequence"] for row in selected}, {f"{index:03d}" for index in range(9)})
        self.assertTrue(all(row["split"] == "train" for row in selected))

    def test_test_leakage_fails_closed(self):
        value = manifest()
        train = next(row for row in value["images"] if row["split"] == "train")
        test = next(row for row in value["images"] if row["split"] == "test")
        test["source_sha256"] = train["source_sha256"]
        with self.assertRaises(ValueError):
            frozen_train_images(value)

    def test_canonical_target_strips_review_only_metadata(self):
        target = canonical_teacher_target(discovery(), ["pallet", "person"])
        self.assertEqual(list(target["assessments"]), ["pallet", "person"])
        self.assertNotIn("label", target["assessments"]["pallet"])
        self.assertEqual(target["unknown_candidates"], [{
            "candidate_name": "barrel", "evidence": "blue cylinder visible", "confidence": 0.7}])

    def test_teacher_record_is_candidate_not_truth_or_training_approval(self):
        source = b"source"
        image = manifest()["images"][0]
        image["source_sha256"] = hashlib.sha256(source).hexdigest()
        raw = {"assessments": {}, "unknown_candidates": []}
        trace = {
            "service": "GPT", "status": "succeeded", "input": {"model": "teacher8b"},
            "output": {"raw_text": json.dumps(raw), "parsed": raw, "finish_reason": "stop",
                       "model_route": "primary", "busy_retry_count": 0,
                       "local_vlm": {"cache_hit": False, "coalesced": False,
                                     "inference_stats": {"completion_tokens": 20}}},
        }
        record = teacher_candidate_record(
            image, source, b"normalized", discovery(), trace, ["pallet", "person"],
            "state", "contract", "model-config")
        self.assertTrue(record["distillation_candidate"])
        self.assertFalse(record["human_truth"])
        self.assertFalse(record["training_authorized"])
        self.assertTrue(record["requires_acceptance_gate"])
        self.assertEqual(record["teacher"]["raw_parsed"], raw)
        self.assertFalse(record["teacher"]["cache_hit"])

    def test_consistency_flag_requires_review(self):
        source = b"source"
        image = manifest()["images"][0]
        image["source_sha256"] = hashlib.sha256(source).hexdigest()
        trace = {
            "service": "GPT", "status": "succeeded", "input": {"model": "teacher8b"},
            "output": {"raw_text": "{}", "parsed": {}, "finish_reason": "stop", "model_route": "primary",
                       "local_vlm": {"cache_hit": False, "coalesced": False}},
        }
        value = teacher_candidate_record(
            image, source, b"normalized", discovery([{"reason_code": "conflict"}]), trace,
            ["pallet", "person"], "state", "contract", "model-config")
        self.assertFalse(value["distillation_candidate"])


if __name__ == "__main__":
    unittest.main()
