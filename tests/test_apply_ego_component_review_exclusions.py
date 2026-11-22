import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tests import test_build_ego_shadow_component_review as review_tests
from tools.apply_ego_component_review_exclusions import apply_exclusions
from tools.build_ego_component_union_review import file_sha256
from tools.build_ego_shadow_component_review import build_component_review


class EgoComponentReviewExclusionTests(unittest.TestCase):
    def _fixture(self, root: Path):
        helper = review_tests.EgoShadowComponentReviewTests()
        queue, identity, shadow = helper._fixture(root)
        review_dir = root / "review"
        review = build_component_review([queue], identity, shadow, review_dir)
        row = review["rows"][0]
        receipt = root / "receipt.json"
        receipt.write_text(json.dumps({
            "candidate_only": True,
            "human_truth": False,
            "approved_for_training": False,
            "production_deployed": False,
            "default_enabled": False,
            "source_component_review_packet_sha256": file_sha256(
                review_dir / "component_review.json"),
            "decisions": [{
                "component_id": row["component_id"],
                "area_pixels": row["area_pixels"],
                "decision": "uncertain",
                "review_image_sha256": row["review_image_sha256"],
            }],
        }), encoding="utf-8")
        return shadow, review_dir / "component_review.json", receipt

    def test_uncertain_component_is_reverted_without_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shadow, review, receipt = self._fixture(root)
            packet = apply_exclusions(
                shadow, review, receipt, root / "refined")
            self.assertEqual(packet["excluded_component_count"], 1)
            self.assertEqual(packet["excluded_pixel_total"], 1)
            self.assertEqual(packet["remaining_candidate_added_pixel_total"], 0)
            self.assertFalse(packet["approved_for_training"])
            self.assertFalse(packet["default_enabled"])
            source = json.loads(shadow.read_text(encoding="utf-8"))
            self.assertEqual(packet["identity_manifest_sha256"],
                             source["identity_manifest_sha256"])

    def test_tampered_decision_binding_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shadow, review, receipt = self._fixture(root)
            packet = json.loads(receipt.read_text(encoding="utf-8"))
            packet["decisions"][0]["area_pixels"] = 2
            receipt.write_text(json.dumps(packet), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "decision binding mismatch"):
                apply_exclusions(shadow, review, receipt, root / "refined")

    def test_write_failure_is_atomic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shadow, review, receipt = self._fixture(root)
            with mock.patch(
                    "tools.apply_ego_component_review_exclusions._save_mask",
                    side_effect=OSError("simulated write failure")):
                with self.assertRaisesRegex(OSError, "simulated write failure"):
                    apply_exclusions(
                        shadow, review, receipt, root / "refined")
            self.assertFalse((root / "refined").exists())
            self.assertEqual(list(root.glob(".refined.staging-*")), [])


if __name__ == "__main__":
    unittest.main()
