import hashlib
import json
from pathlib import Path
import unittest

from source.agent.legacy_evidence_reader import LegacyEvidenceError, read_legacy_evidence


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "schemas/historical-evidence-superseding-correction.schema.json"
MANIFEST_SCHEMA = ROOT / "schemas/historical-evidence-restoration-manifest.schema.json"
MANIFEST = (ROOT / "docs/test-results"
            / "historical-evidence-exact-restoration-corrections-20261111.json")
SIDECAR = (ROOT.parents[1] / "outputs/sfl_cpd15y_stage2_20261110/slope_review"
           / "vlm_slope_candidate_review.json")
ENTRIES = (
    ("docs/test-results/closed-loop-runtime-review-remediation-20261111.md", "35fdf1b09c5e3cf9df629ecc95e4c9d3903586102e6088bbbc44252ba9adcf22", "closed-loop-runtime-review-remediation-20261111.superseding-correction.json"),
    ("docs/test-results/closed-loop-slope-known-label-candidate-generated-20261111.json", "fe801bed6cd617cd3e545c159af14d975d71f8d1f3d3eee1d1076112c5909b00", "closed-loop-slope-known-label-candidate-generated-20261111.superseding-correction.json"),
    ("docs/test-results/closed-loop-slope-known-label-events-20261111.md", "f3d2db0a13a819c919a29454b479c69918d052d448d4c74e8ff29b1216985861", "closed-loop-slope-known-label-events-20261111.superseding-correction.json"),
    ("docs/test-results/closed-loop-slope-known-label-vlm-reviewed-20261111.json", "7cc56fbe1c9493c5dadde5d0d8c0518cf585250e945ccbcad2ae6fc278b2c93e", "closed-loop-slope-known-label-vlm-reviewed-20261111.superseding-correction.json"),
    ("docs/test-results/dual-5090-acceptance-20261111.md", "036de2d944539dca2ffc22b3195e1136875d852d43d42fd630f2ecf3b450b30a", "dual-5090-acceptance-20261111.superseding-correction.json"),
    ("docs/test-results/dual-5090-packaging-newline-defect-20261111.md", "2907fd6af91b33fe040ca2bae25e8e415d257b42fe664e919ee1f20a1c0f2367", "dual-5090-packaging-newline-defect-20261111.superseding-correction.json"),
    ("docs/test-results/fixed-holdout-vertical-chain-receipt-20261111.json", "1caa15b53e573301928288f9ff9a459619a18339948d0cef146dc8c4f30200eb", "fixed-holdout-vertical-chain-receipt-20261111.superseding-correction.json"),
    ("docs/test-results/fixed-holdout-vertical-chain-receipt-20261111.md", "70c4cc4029e9e5668ee9c7cfeadeedf4559fa1a5b09b2d5d878ba59053bb065e", "fixed-holdout-vertical-chain-receipt-20261111.md.superseding-correction.json"),
    ("docs/test-results/local-py311-human-review-slope-verification-20261111.json", "9626d26de1a5d8fad1c7f2acdb39b789f394a8371215dda7361c46e4817c5cc5", "local-py311-human-review-slope-verification-20261111.superseding-correction.json"),
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
        cls.manifest_validator = Draft202012Validator(
            json.loads(MANIFEST_SCHEMA.read_text(encoding="utf-8")),
            format_checker=FormatChecker(),
        )

    def test_all_eleven_historical_files_match_cc20fb0_sha(self):
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

    def test_total_manifest_is_schema_valid_and_binds_every_restored_artifact(self):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(list(self.manifest_validator.iter_errors(manifest)), [])
        self.assertEqual(manifest["restored_artifact_count"], len(ENTRIES))
        prior_receipt = manifest["supersedes_prior_restore_receipt"]
        prior_path = ROOT / prior_receipt["path"]
        self.assertFalse(prior_receipt["prior_artifact_modified"])
        self.assertEqual(hashlib.sha256(prior_path.read_bytes()).hexdigest(),
                         prior_receipt["sha256"])
        self.assertEqual(prior_receipt["prior_restored_file_count"], 9)
        self.assertEqual(prior_receipt["corrected_restored_file_count"], 11)
        records = {entry["path"]: entry for entry in manifest["restored_artifacts"]}
        self.assertEqual(set(records), {entry[0] for entry in ENTRIES})
        for relative, expected_sha, correction_name in ENTRIES:
            record = records[relative]
            correction_path = ROOT / "docs/test-results" / correction_name
            correction = json.loads(correction_path.read_text(encoding="utf-8"))
            with self.subTest(path=relative):
                self.assertTrue(record["exact_blob_restored"])
                self.assertEqual(record["sha256"], expected_sha)
                self.assertEqual(record["correction_event_path"],
                                 f"docs/test-results/{correction_name}")
                self.assertEqual(
                    record["correction_event_sha256"],
                    hashlib.sha256(correction_path.read_bytes()).hexdigest(),
                )
                self.assertEqual(record["correction_event_id"],
                                 correction["correction_id"])

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
