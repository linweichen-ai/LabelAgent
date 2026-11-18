import unittest

from tools.fuse_slope_candidate_evidence import fuse


def fixtures():
    visible = [False, False, True, True, False, True, True, False]
    vlm_present = {3, 6}
    vlm = {
        "schema_version": "labelagent.qwen3vl-slope-review-score.v1",
        "human_truth": False, "human_confirmed": False,
        "production_deployed": False,
        "summary": {"details": [
            {"sample_id": f"s{i}",
             "assistant_visible_slope_candidate": truth,
             "candidate_evidence": "present" if i in vlm_present else "absent"}
            for i, truth in enumerate(visible)]},
    }
    geometry = {
        "schema_version": "labelagent.slope-lidar-geometry.v1",
        "human_truth": False, "human_confirmed": False,
        "production_deployed": False,
        "images": [
            {"sample_id": f"s{i}", "image_sha256": f"h{i}",
             "generated_hint_count": 1 if truth else 0}
            for i, truth in enumerate(visible)],
    }
    sam = {
        "schema_version": "labelagent.slope-lidar-sam.v1",
        "human_truth": False, "human_confirmed": False,
        "production_deployed": False,
        "rows": [
            {"sample_id": f"s{i}", "image_sha256": f"h{i}",
             "mask_pixels": 10}
            for i, truth in enumerate(visible) if truth],
    }
    return vlm, geometry, sam


class FuseSlopeCandidateEvidenceTests(unittest.TestCase):
    def test_geometry_sam_rescues_bounded_vlm_misses(self):
        result = fuse(*fixtures())
        self.assertEqual(result["true_positive_candidates"], 4)
        self.assertEqual(result["false_positive_candidates"], 0)
        self.assertEqual(result["false_negative_candidates"], 0)
        self.assertEqual(result["recall_on_assistant_review"], 1.0)

    def test_hint_without_sam_fails_closed(self):
        vlm, geometry, sam = fixtures()
        sam["rows"] = sam["rows"][1:]
        with self.assertRaisesRegex(ValueError, "no matching SAM"):
            fuse(vlm, geometry, sam)

    def test_hash_mismatch_fails_closed(self):
        vlm, geometry, sam = fixtures()
        sam["rows"][0]["image_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "hashes differ"):
            fuse(vlm, geometry, sam)

    def test_machine_evidence_cannot_claim_human_truth(self):
        vlm, geometry, sam = fixtures()
        geometry["human_truth"] = True
        with self.assertRaisesRegex(ValueError, "unsafe"):
            fuse(vlm, geometry, sam)


if __name__ == "__main__":
    unittest.main()
