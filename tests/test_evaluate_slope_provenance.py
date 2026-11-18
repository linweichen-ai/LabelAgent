import unittest

from source.agent.evaluation_provenance import classify_target_provenance


def shape(source=None, **flags):
    value = {"label": "slope", "points": [[0, 0], [1, 0], [1, 1]]}
    if source is not None:
        value["cvat_metadata"] = {"source": source}
    value.update(flags)
    return value


def verified_status():
    return {
        "current_human_review_verified": True,
        "reusable_as_reference": True,
        "reason_code": "verified_current_human_truth",
        "checks": {
            "nonempty_truth": True,
            "current_contract": True,
            "truth_sha256_matches": True,
            "transaction_committed": True,
        },
    }


class SlopeTargetProvenanceTests(unittest.TestCase):
    def test_legacy_auto_shapes_are_not_human_truth(self):
        result = classify_target_provenance(
            {"truth": {"74": [shape("auto")], "77": [shape("auto")]}},
            [74, 77], ["slope"])
        self.assertEqual(result["kind"], "legacy_auto_preannotation")
        self.assertFalse(result["human_truth_verified"])
        self.assertEqual(result["source_counts"]["auto"], 2)
        self.assertEqual(result["unverified_frames"], [74, 77])

    def test_storage_under_truth_is_not_sufficient(self):
        result = classify_target_provenance(
            {"truth": {"3": [shape("manual")]}}, [3], ["slope"])
        self.assertEqual(result["kind"], "unverified_saved_annotations")
        self.assertFalse(result["human_truth_verified"])

    def test_explicit_machine_boundary_takes_precedence(self):
        result = classify_target_provenance(
            {"truth": {"4": [shape("auto", candidate_only=True,
                                    human_truth=False)]}}, [4], ["slope"])
        self.assertEqual(result["kind"], "machine_candidate")
        self.assertEqual(result["boundary_counts"]["candidate_only_true"], 1)

    def test_all_frames_need_current_committed_receipts(self):
        session = {
            "truth": {"5": [shape("auto")], "6": [shape("auto")]},
            "truth_review_status": {"5": verified_status()},
        }
        result = classify_target_provenance(session, [5, 6], ["slope"])
        self.assertFalse(result["human_truth_verified"])
        self.assertEqual(result["verified_frames"], [5])
        self.assertEqual(result["unverified_frames"], [6])

    def test_review_receipts_override_legacy_source_metadata(self):
        session = {
            "truth": {"5": [shape("auto")], "6": [shape("auto")]},
            "truth_review_status": {
                "5": verified_status(), "6": verified_status(),
            },
        }
        result = classify_target_provenance(session, [5, 6], ["slope"])
        self.assertEqual(result["kind"], "human_reviewed")
        self.assertTrue(result["human_truth_verified"])
        self.assertEqual(result["unverified_frames"], [])

    def test_incomplete_review_checks_fail_closed(self):
        incomplete = verified_status()
        incomplete["checks"]["transaction_committed"] = False
        session = {
            "truth": {"8": [shape("manual")]},
            "truth_review_status": {"8": incomplete},
        }
        result = classify_target_provenance(session, [8], ["slope"])
        self.assertEqual(result["kind"], "unverified_saved_annotations")
        self.assertFalse(result["human_truth_verified"])


if __name__ == "__main__":
    unittest.main()
