import unittest

from tools.build_ego_component_risk_blind_audit import (
    _band_for_camera_rank, stratified_sample)


class EgoComponentRiskBlindAuditTests(unittest.TestCase):
    def test_band_boundaries_cover_top_middle_bottom(self):
        self.assertEqual(_band_for_camera_rank(1, 100), "top")
        self.assertEqual(_band_for_camera_rank(50, 100), "middle")
        self.assertEqual(_band_for_camera_rank(100, 100), "bottom")
        self.assertIsNone(_band_for_camera_rank(30, 100))

    def test_stratified_sample_is_balanced_blinded_and_deterministic(self):
        rows = []
        for camera in ("camera1", "camera3"):
            for rank in range(1, 101):
                rows.append({
                    "component_id": f"{camera}-{rank:03d}",
                    "camera": camera,
                    "camera_review_priority_rank": rank,
                    "review_priority_rank": rank,
                    "review_priority_score": 1.0 - rank / 1000.0,
                })
        first = stratified_sample(rows, 5, "seed")
        second = stratified_sample(rows, 5, "seed")
        self.assertEqual(first, second)
        self.assertEqual(len(first), 30)
        counts = {}
        for row in first:
            counts[(row["audit_camera"], row["audit_band"])] = (
                counts.get((row["audit_camera"], row["audit_band"]), 0) + 1)
            self.assertRegex(row["audit_id"], r"^A\d{3}$")
        self.assertTrue(all(value == 5 for value in counts.values()))

    def test_rejects_empty_seed(self):
        with self.assertRaisesRegex(ValueError, "sampling settings"):
            stratified_sample([], 1, "")

    def test_excluded_components_never_reappear(self):
        rows = []
        for camera in ("camera1", "camera3"):
            for rank in range(1, 101):
                rows.append({
                    "component_id": f"{camera}-{rank:03d}",
                    "camera": camera,
                    "camera_review_priority_rank": rank,
                    "review_priority_rank": rank,
                    "review_priority_score": 1.0 - rank / 1000.0,
                })
        first = stratified_sample(rows, 5, "first")
        excluded = {row["component_id"] for row in first}
        second = stratified_sample(rows, 5, "second", excluded)
        self.assertTrue(excluded.isdisjoint(
            {row["component_id"] for row in second}))
        self.assertEqual(len(second), 30)

    def test_exclusion_fails_closed_when_stratum_is_exhausted(self):
        rows = []
        for camera in ("camera1", "camera3"):
            for rank in range(1, 11):
                rows.append({
                    "component_id": f"{camera}-{rank:03d}",
                    "camera": camera,
                    "camera_review_priority_rank": rank,
                    "review_priority_rank": rank,
                    "review_priority_score": 1.0,
                })
        excluded = {"camera1-001", "camera1-002"}
        with self.assertRaisesRegex(ValueError, "insufficient camera1/top"):
            stratified_sample(rows, 1, "seed", excluded)


if __name__ == "__main__":
    unittest.main()
