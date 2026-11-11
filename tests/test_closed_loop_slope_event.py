import hashlib
import json
from pathlib import Path
import unittest

from source.agent.closed_loop_contract import validate_event
from source.agent.legacy_slope_reader import (
    LegacySlopeArtifactError,
    read_legacy_slope_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "schemas" / "labelagent_closed_loop_event.schema.json"
CORRECTION_SCHEMA = ROOT / "schemas" / "legacy-slope-policy-correction.schema.json"
OLD_ARTIFACTS = (
    (
        ROOT / "docs/test-results/closed-loop-slope-known-label-candidate-generated-20261111.json",
        "fe801bed6cd617cd3e545c159af14d975d71f8d1f3d3eee1d1076112c5909b00",
        "evt-slope-a000-f0-candidate-b6652e98",
        ROOT / "docs/test-results/closed-loop-slope-known-label-candidate-generated-20261111.policy-correction.json",
    ),
    (
        ROOT / "docs/test-results/closed-loop-slope-known-label-vlm-reviewed-20261111.json",
        "7cc56fbe1c9493c5dadde5d0d8c0518cf585250e945ccbcad2ae6fc278b2c93e",
        "evt-slope-a000-f0-vlm-aee398ed",
        ROOT / "docs/test-results/closed-loop-slope-known-label-vlm-reviewed-20261111.policy-correction.json",
    ),
    (
        ROOT / "docs/test-results/slope-frame0-overlay-review-20261111.json",
        "79a1be7b9e0a7141a1233e6cc59572e9b42abe63b19312345c9e37db478c7a19",
        None,
        ROOT / "docs/test-results/slope-frame0-overlay-review-20261111.policy-correction.json",
    ),
)
SIDECAR = (ROOT.parents[1] / "outputs/sfl_cpd15y_stage2_20261110/slope_review"
           / "vlm_slope_candidate_review.json")
SIDECAR_SHA = "aee398edace4c7b46ed04a1379a05cd6f0502e3809b7ae1a88972dba005a7dea"
SIDECAR_CORRECTION = (ROOT / "docs/test-results"
                      / "vlm-slope-sidecar-legacy-policy-correction-20261111.json")


class ClosedLoopSlopeLegacyBoundaryTests(unittest.TestCase):
    @staticmethod
    def _sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_old_artifact_bytes_and_hashes_are_unchanged(self):
        for path, expected_sha, _, _ in OLD_ARTIFACTS:
            with self.subTest(path=path.name):
                self.assertEqual(self._sha(path), expected_sha)
        self.assertEqual(self._sha(SIDECAR), SIDECAR_SHA)

    def test_archival_reader_accepts_old_bytes_but_cannot_emit_current_event(self):
        for path, expected_sha, old_event_id, _ in OLD_ARTIFACTS:
            with self.subTest(path=path.name):
                snapshot = read_legacy_slope_artifact(
                    path,
                    expected_sha256=expected_sha,
                    legacy_identifier=f"legacy:{path.stem}",
                )
                self.assertTrue(snapshot.archival_only)
                self.assertFalse(snapshot.current_ingest_allowed)
                self.assertEqual(snapshot.old_event_id, old_event_id)
                self.assertIn("traversability_status", snapshot.legacy_fields)
                with self.assertRaisesRegex(
                    LegacySlopeArtifactError,
                    "cannot be emitted as current events",
                ):
                    snapshot.as_current_event()

        sidecar = read_legacy_slope_artifact(
            SIDECAR,
            expected_sha256=SIDECAR_SHA,
            legacy_identifier="legacy:vlm_slope_candidate_review",
        )
        self.assertTrue(sidecar.archival_only)
        self.assertFalse(sidecar.current_ingest_allowed)

    def test_archival_reader_rejects_hash_mismatch(self):
        path, _, _, _ = OLD_ARTIFACTS[0]
        with self.assertRaisesRegex(LegacySlopeArtifactError, "SHA-256 mismatch"):
            read_legacy_slope_artifact(path, expected_sha256="0" * 64)

    def test_old_transport_events_are_rejected_by_current_runtime(self):
        for path, _, _, _ in OLD_ARTIFACTS[:2]:
            event = json.loads(path.read_text(encoding="utf-8"))
            errors = validate_event(event)
            with self.subTest(path=path.name, errors=errors):
                self.assertIn("slope_context.label_type must be slope", errors)
                self.assertIn(
                    "slope geometry evidence cannot gate the default passable mapping",
                    errors,
                )
                self.assertIn("slope policy_default must be passable", errors)
                self.assertIn("slope policy_override is required", errors)

    def test_old_transport_events_fail_current_schema(self):
        try:
            from jsonschema import Draft202012Validator, FormatChecker
        except ImportError:
            self.skipTest("jsonschema development dependency is absent")
        validator = Draft202012Validator(
            json.loads(SCHEMA.read_text(encoding="utf-8")),
            format_checker=FormatChecker(),
        )
        for path, _, _, _ in OLD_ARTIFACTS[:2]:
            event = json.loads(path.read_text(encoding="utf-8"))
            with self.subTest(path=path.name):
                self.assertTrue(list(validator.iter_errors(event)))

    def test_corrections_validate_as_exact_files_and_bind_old_artifacts(self):
        try:
            from jsonschema import Draft202012Validator, FormatChecker
        except ImportError:
            self.skipTest("jsonschema development dependency is absent")
        validator = Draft202012Validator(
            json.loads(CORRECTION_SCHEMA.read_text(encoding="utf-8")),
            format_checker=FormatChecker(),
        )
        for old_path, expected_sha, old_event_id, correction_path in OLD_ARTIFACTS:
            correction = json.loads(correction_path.read_text(encoding="utf-8"))
            with self.subTest(path=correction_path.name):
                self.assertEqual(list(validator.iter_errors(correction)), [])
                binding = correction["archival_binding"]
                self.assertEqual(binding["old_artifact_sha256"], expected_sha)
                self.assertEqual(binding["old_event_id"], old_event_id)
                self.assertEqual(binding["old_artifact_path"], old_path.relative_to(ROOT).as_posix())
                self.assertEqual(correction["superseded_by"], correction["correction_id"])
                self.assertTrue(binding["archival_only"])
                self.assertFalse(binding["current_ingest_allowed"])

        sidecar_correction = json.loads(SIDECAR_CORRECTION.read_text(encoding="utf-8"))
        self.assertEqual(list(validator.iter_errors(sidecar_correction)), [])
        sidecar_binding = sidecar_correction["archival_binding"]
        self.assertEqual(sidecar_binding["old_artifact_sha256"], SIDECAR_SHA)
        self.assertIsNone(sidecar_binding["old_event_id"])
        self.assertEqual(
            sidecar_binding["old_artifact_path"],
            "outputs/sfl_cpd15y_stage2_20261110/slope_review/vlm_slope_candidate_review.json",
        )
        self.assertEqual(
            sidecar_correction["superseded_by"],
            sidecar_correction["correction_id"],
        )

    def test_corrections_contain_only_current_passable_policy(self):
        correction_paths = [item[3] for item in OLD_ARTIFACTS] + [SIDECAR_CORRECTION]
        for correction_path in correction_paths:
            correction = json.loads(correction_path.read_text(encoding="utf-8"))
            encoded = json.dumps(correction, sort_keys=True)
            with self.subTest(path=correction_path.name):
                self.assertNotIn("geometry_evidence_sufficient", encoded)
                self.assertNotIn("geometry_evidence_artifact_ids", encoded)
                policy = correction["new_policy"]
                self.assertEqual(policy["traversability_status"], "passable")
                self.assertEqual(policy["policy_basis"], "user_defined_project_semantics")
                self.assertEqual(policy["downstream_mapping"]["source_labels"], ["floor", "slope"])
                self.assertFalse(policy["human_pixel_truth_available"])
                self.assertFalse(policy["approved_for_training"])


if __name__ == "__main__":
    unittest.main()
