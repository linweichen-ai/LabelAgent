import hashlib
import json
from pathlib import Path
import unittest

from source.agent.legacy_evidence_reader import LegacyEvidenceError, read_legacy_evidence


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "schemas/historical-evidence-superseding-correction.schema.json"
SIDECAR = (ROOT.parents[1] / "outputs/sfl_cpd15y_stage2_20261110/slope_review"
           / "vlm_slope_candidate_review.json")
ENTRIES = (
    ("docs/test-results/closed-loop-slope-known-label-candidate-generated-20261111.json", "fe801bed6cd617cd3e545c159af14d975d71f8d1f3d3eee1d1076112c5909b00", "closed-loop-slope-known-label-candidate-generated-20261111.superseding-correction.json"),
    ("docs/test-results/closed-loop-slope-known-label-events-20261111.md", "f3d2db0a13a819c919a29454b479c69918d052d448d4c74e8ff29b1216985861", "closed-loop-slope-known-label-events-20261111.superseding-correction.json"),
    ("docs/test-results/closed-loop-slope-known-label-vlm-reviewed-20261111.json", "7cc56fbe1c9493c5dadde5d0d8c0518cf585250e945ccbcad2ae6fc278b2c93e", "closed-loop-slope-known-label-vlm-reviewed-20261111.superseding-correction.json"),
    ("docs/test-results/dual-5090-acceptance-20261111.md", "036de2d944539dca2ffc22b3195e1136875d852d43d42fd630f2ecf3b450b30a", "dual-5090-acceptance-20261111.superseding-correction.json"),
    ("docs/test-results/dual-5090-packaging-newline-defect-20261111.md", "2907fd6af91b33fe040ca2bae25e8e415d257b42fe664e919ee1f20a1c0f2367", "dual-5090-packaging-newline-defect-20261111.superseding-correction.json"),
    ("docs/test-results/fixed-holdout-vertical-chain-receipt-20261111.json", "1caa15b53e573301928288f9ff9a459619a18339948d0cef146dc8c4f30200eb", "fixed-holdout-vertical-chain-receipt-20261111.superseding-correction.json"),
    ("docs/test-results/fixed-holdout-vertical-chain-receipt-20261111.md", "70c4cc4029e9e5668ee9c7cfeadeedf4559fa1a5b09b2d5d878ba59053bb065e", "fixed-holdout-vertical-chain-receipt-20261111.md.superseding-correction.json"),
    ("docs/test-results/slope-frame0-overlay-review-20261111.json", "79a1be7b9e0a7141a1233e6cc59572e9b42abe63b19312345c9e37db478c7a19", "slope-frame0-overlay-review-20261111.superseding-correction.json"),
    ("docs/test-results/slope-frame0-overlay-review-20261111.md", "1b93bd1ae0963141202bb1133dfd626a9b3b690283126d04780e34908a037fce", "slope-frame0-overlay-review-20261111.md.superseding-correction.json"),
)
SIDECAR_CORRECTION = ROOT / "docs/test-results/vlm-slope-candidate-review-sidecar-20261111.superseding-correction.json"


class HistoricalEvidenceCorrectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from jsonschema import Draft202012Validator, FormatChecker
        except ImportError as error:
            raise unittest.SkipTest("jsonschema development dependency is absent") from error
        cls.validator = Draft202012Validator(
            json.loads(SCHEMA.read_text(encoding="utf-8")),
            format_checker=FormatChecker(),
        )

    def test_all_nine_963a46c_historical_files_match_cc20fb0_sha(self):
        for relative, expected_sha, _ in ENTRIES:
            path = ROOT / relative
            with self.subTest(path=relative):
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected_sha)

    def test_each_historical_file_has_schema_valid_superseding_correction(self):
        for relative, expected_sha, correction_name in ENTRIES:
            correction_path = ROOT / "docs/test-results" / correction_name
            correction = json.loads(correction_path.read_text(encoding="utf-8"))
            with self.subTest(path=correction_name):
                self.assertEqual(list(self.validator.iter_errors(correction)), [])
                self.assertEqual(correction["prior_artifact_path"], relative)
                self.assertEqual(correction["prior_sha256"], expected_sha)
                self.assertEqual(correction["superseded_by"], correction["correction_id"])
                self.assertFalse(correction["new_policy"]["human_pixel_truth_available"])
                self.assertFalse(correction["new_policy"]["approved_for_training"])
                encoded = json.dumps(correction, sort_keys=True)
                self.assertNotIn("geometry_evidence_sufficient", encoded)
                self.assertNotIn('"traversability_status": "unknown"', encoded)

    def test_fixed_holdout_is_routed_after_candidate_final(self):
        names = (
            "fixed-holdout-vertical-chain-receipt-20261111.superseding-correction.json",
            "fixed-holdout-vertical-chain-receipt-20261111.md.superseding-correction.json",
        )
        expected = {
            "from_state": "candidate_final",
            "to_state": "human_review_required",
            "reason": "human_pixel_truth_missing",
        }
        for name in names:
            correction = json.loads((ROOT / "docs/test-results" / name).read_text(encoding="utf-8"))
            self.assertEqual(correction["routing_correction"], expected)

    def test_sidecar_is_unchanged_and_has_schema_valid_correction(self):
        expected_sha = "aee398edace4c7b46ed04a1379a05cd6f0502e3809b7ae1a88972dba005a7dea"
        self.assertEqual(hashlib.sha256(SIDECAR.read_bytes()).hexdigest(), expected_sha)
        correction = json.loads(SIDECAR_CORRECTION.read_text(encoding="utf-8"))
        self.assertEqual(list(self.validator.iter_errors(correction)), [])
        self.assertEqual(correction["prior_sha256"], expected_sha)

    def test_archival_reader_is_hash_bound_and_cannot_emit_current_events(self):
        for relative, expected_sha, correction_name in ENTRIES:
            correction = json.loads((ROOT / "docs/test-results" / correction_name).read_text(encoding="utf-8"))
            snapshot = read_legacy_evidence(
                ROOT / relative,
                expected_sha256=expected_sha,
                prior_event_id=correction["prior_event_id"],
            )
            self.assertTrue(snapshot.archival_only)
            self.assertFalse(snapshot.current_ingest_allowed)
            with self.assertRaises(LegacyEvidenceError):
                snapshot.as_current_event()


if __name__ == "__main__":
    unittest.main()
