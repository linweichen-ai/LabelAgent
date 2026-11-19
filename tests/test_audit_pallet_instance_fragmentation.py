import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tools.audit_pallet_instance_fragmentation import (
    rank_disjoint_pair_controls,
    rank_fragment_pairs,
    render_fragment_review,
)


def rectangle(shape_id, left, top, right, bottom):
    points = [[left, top], [right, top], [right, bottom], [left, bottom]]
    return {"id": shape_id, "label": "pallet", "points": points,
            "holes": [], "parts": [{"points": points, "holes": []}]}


class PalletInstanceFragmentationAuditTest(unittest.TestCase):
    def test_ranking_prioritizes_nested_small_fragment(self):
        large = rectangle("large", 10, 10, 130, 80)
        nested = rectangle("nested", 20, 20, 50, 35)
        adjacent_peer = rectangle("peer", 135, 10, 250, 80)
        payload = {"/capture/camera_0/image": {
            "0": [large, nested, adjacent_peer]}}

        rows, considered, hypothesis_count, conservative_count = rank_fragment_pairs(
            payload, limit=10)

        self.assertEqual(considered, 3)
        self.assertEqual(hypothesis_count, 1)
        self.assertEqual(conservative_count, 1)
        self.assertEqual(rows[0]["small_id"], "nested")
        self.assertEqual(rows[0]["large_id"], "large")
        self.assertEqual(
            rows[0]["relation_hypothesis"],
            "possible_duplicate_or_nested_fragment")
        self.assertFalse(rows[0]["automatic_merge_allowed"])
        self.assertTrue(rows[0]["conservative_duplicate_gate"])

    def test_equal_adjacent_instances_are_not_merge_hypotheses(self):
        first = rectangle("first", 10, 10, 100, 70)
        second = rectangle("second", 105, 10, 195, 70)
        payload = {"/capture/camera_0/image": {"0": [first, second]}}

        rows, considered, hypothesis_count, conservative_count = rank_fragment_pairs(
            payload, limit=10)

        self.assertEqual(considered, 1)
        self.assertEqual(hypothesis_count, 0)
        self.assertEqual(conservative_count, 0)
        self.assertEqual(rows, [])

        controls, control_considered, control_count = rank_disjoint_pair_controls(
            payload, limit=10)
        self.assertEqual(control_considered, 1)
        self.assertEqual(control_count, 1)
        self.assertEqual(controls[0]["relation_hypothesis"],
                         "disjoint_pair_control_candidate")

    def test_conservative_only_filters_review_hypotheses_before_limit(self):
        large = rectangle("large", 10, 10, 130, 80)
        nested = rectangle("nested", 20, 20, 50, 35)
        partial = rectangle("partial", 5, 20, 60, 40)
        payload = {"/capture/camera_0/image": {"0": [large, nested, partial]}}

        rows, _, hypothesis_count, conservative_count = rank_fragment_pairs(
            payload, limit=10, conservative_only=True)

        self.assertGreaterEqual(hypothesis_count, conservative_count)
        self.assertTrue(rows)
        self.assertTrue(all(row["conservative_duplicate_gate"] for row in rows))

    def test_render_is_read_only_and_never_approves_merge(self):
        large = rectangle("sam3-large", 10, 10, 130, 80)
        nested = rectangle("sam3-fragment", 20, 20, 50, 35)
        payload = {"/capture/camera_0/image": {"0": [large, nested]}}
        audit = {"results": [
            {"topic": "/capture/camera_0/image", "frame": 0,
             "shape_id": "sam3-large", "pixel_area": 8591},
            {"topic": "/capture/camera_0/image", "frame": 0,
             "shape_id": "sam3-fragment", "pixel_area": 496},
        ]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = root / "candidates.json"
            candidates.write_text(json.dumps(payload), encoding="utf-8")
            before = candidates.read_bytes()
            audit_path = root / "audit.json"
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            camera = root / "images" / "camera_0"
            camera.mkdir(parents=True)
            Image.new("RGB", (180, 110), "gray").save(camera / "000.png")

            report = render_fragment_review(
                candidates, root / "images", audit_path, root / "output",
                limit=1)

            self.assertEqual(candidates.read_bytes(), before)
            self.assertEqual(report["selected_count"], 1)
            self.assertFalse(report["automatic_merge_allowed"])
            self.assertTrue(report["rows"][0]["review_required"])
            self.assertTrue(Path(report["contact_sheet"]).is_file())
            self.assertTrue(Path(report["report_path"]).is_file())

    def test_render_accepts_flat_numeric_frame_jpg_without_audit(self):
        payload = {"0": [
            rectangle("large", 10, 10, 130, 80),
            rectangle("nested", 20, 20, 50, 35),
        ]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = root / "candidates.json"
            candidates.write_text(json.dumps(payload), encoding="utf-8")
            images = root / "images"
            images.mkdir()
            Image.new("RGB", (180, 110), "gray").save(images / "frame_000.jpg")

            report = render_fragment_review(
                candidates, images, None, root / "output", limit=1)

            self.assertEqual(report["selected_count"], 1)
            self.assertIsNone(report["audit_report"])
            self.assertIn("/CVAT/Task3/images", report["rows"][0]["topic"])


if __name__ == "__main__":
    unittest.main()
