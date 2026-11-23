"""Tests for the bounded external expert review contract."""
import copy
import unittest

from source.agent.expert_review import (
    build_expert_review_instruction, expert_review_schema,
    review_external_candidate_shadow, validate_expert_review,
)


def candidates():
    return [{"id": "c1", "label": "pallet", "points": [[1, 2]]},
            {"id": "c2", "label": "slope", "points": [[3, 4]]}]


def valid_result():
    return {
        "overall_decision": "review_required",
        "candidate_reviews": [
            {"candidate_id": "c1", "label": "pallet",
             "verdict": "boundary_issue", "confidence": 0.91,
             "reason": "Outline includes visible floor pixels.",
             "suggested_label": None},
            {"candidate_id": "c2", "label": "slope",
             "verdict": "supported", "confidence": 0.87,
             "reason": "Ramp surface is visibly covered.",
             "suggested_label": None},
        ],
        "missing_labels": [],
        "new_label_candidates": [{"name": "rack", "confidence": 0.72,
                                  "evidence": "A metal rack is visible."}],
    }


class ExpertReviewTests(unittest.TestCase):
    def test_prompt_binds_ids_without_coordinates_or_edit_operations(self):
        instruction = build_expert_review_instruction(
            candidates(), ["pallet", "slope"],
            routing_reasons=["visibility_uncertain"])
        self.assertIn('"candidate_id": "c1"', instruction)
        self.assertIn("visibility_uncertain", instruction)
        self.assertNotIn("[[1, 2]]", instruction)
        self.assertIn("do not output editing operations", instruction)

    def test_schema_enumerates_exact_candidate_ids_and_labels(self):
        schema = expert_review_schema(["c1", "c2"], ["pallet", "slope"])
        item = schema["properties"]["candidate_reviews"]["items"]
        self.assertEqual(item["properties"]["candidate_id"]["enum"],
                         ["c1", "c2"])
        self.assertEqual(item["properties"]["label"]["enum"],
                         ["pallet", "slope"])
        self.assertEqual(schema["properties"]["candidate_reviews"]["minItems"], 2)

    def test_validation_preserves_assistant_only_boundaries(self):
        normalized = validate_expert_review(
            valid_result(), candidates(), ["pallet", "slope"])
        self.assertEqual(normalized["route"], "assistant_review_only")
        self.assertTrue(normalized["candidate_only"])
        self.assertFalse(normalized["human_truth"])
        self.assertFalse(normalized["approved_for_training"])
        self.assertFalse(normalized["candidate_mutated"])
        self.assertFalse(normalized["ontology_mutated"])

    def test_validation_rejects_missing_duplicate_unknown_and_illegal_suggestion(self):
        missing = valid_result()
        missing["candidate_reviews"].pop()
        with self.assertRaisesRegex(ValueError, "every candidate"):
            validate_expert_review(missing, candidates(), ["pallet", "slope"])
        duplicate = valid_result()
        duplicate["candidate_reviews"][1]["candidate_id"] = "c1"
        duplicate["candidate_reviews"][1]["label"] = "pallet"
        with self.assertRaisesRegex(ValueError, "invalid"):
            validate_expert_review(duplicate, candidates(), ["pallet", "slope"])
        unknown = valid_result()
        unknown["candidate_reviews"][0]["candidate_id"] = "outside"
        with self.assertRaisesRegex(ValueError, "invalid"):
            validate_expert_review(unknown, candidates(), ["pallet", "slope"])
        suggestion = valid_result()
        suggestion["candidate_reviews"][0]["suggested_label"] = "slope"
        with self.assertRaisesRegex(ValueError, "only valid"):
            validate_expert_review(suggestion, candidates(), ["pallet", "slope"])

    def test_new_label_alias_of_known_class_is_rejected(self):
        result = valid_result()
        result["new_label_candidates"][0]["name"] = "Pallet"
        with self.assertRaisesRegex(ValueError, "new-label proposal is invalid"):
            validate_expert_review(result, candidates(), ["pallet", "slope"])

    def test_explicit_shadow_call_uses_two_images_and_never_mutates_candidates(self):
        original_candidates = candidates()
        before = copy.deepcopy(original_candidates)
        observed = {}
        def respond(config, instruction, image, extra_images, response_schema):
            observed.update(config=config, instruction=instruction, image=image,
                            extra_images=extra_images, schema=response_schema)
            return valid_result()
        normalized = review_external_candidate_shadow(
            {"llm_expert_enabled": True}, b"original", b"overlay",
            original_candidates, ["pallet", "slope"],
            routing_reasons=["agent_nonconvergence"], chat_fn=respond)
        self.assertEqual(observed["image"], b"original")
        self.assertEqual(observed["extra_images"], (b"overlay",))
        self.assertEqual(original_candidates, before)
        self.assertFalse(normalized["candidate_mutated"])


if __name__ == "__main__":
    unittest.main()
