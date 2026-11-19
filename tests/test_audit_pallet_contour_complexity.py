import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tools.audit_pallet_contour_complexity import (
    contour_metrics,
    rank_candidates,
    render_review,
)


def rectangle(shape_id, points):
    return {"id": shape_id, "label": "pallet", "points": points,
            "holes": [], "parts": [{"points": points, "holes": []}]}


class PalletContourComplexityAuditTest(unittest.TestCase):
    def test_metrics_count_outer_vertices_and_holes_separately(self):
        shape = rectangle(1, [[10, 10], [30, 10], [30, 30], [10, 30]])
        shape["parts"][0]["holes"] = [[[15, 15], [20, 15], [20, 20], [15, 20]]]

        metrics = contour_metrics(shape)

        self.assertEqual(metrics["outer_vertices"], 4)
        self.assertEqual(metrics["hole_vertices"], 4)
        self.assertEqual(metrics["bbox_area"], 441.0)

    def test_ranking_is_deterministic_and_filters_small_boxes(self):
        smooth = rectangle("smooth", [[0, 0], [100, 0], [100, 50], [0, 50]])
        jagged_points = [[0, 0], [20, 0], [40, 1], [60, 0], [80, 1], [100, 0],
                         [100, 50], [80, 49], [60, 50], [40, 49], [20, 50], [0, 50]]
        jagged = rectangle("jagged", jagged_points)
        tiny = rectangle("tiny", [[0, 0], [5, 0], [5, 5], [0, 5]])
        payload = {"/capture/camera_0/image": {"0": [smooth, tiny], "1": [jagged]}}

        ranked = rank_candidates(payload, minimum_bbox_area=1000, limit=5)

        self.assertEqual([row["shape_id"] for row in ranked], ["jagged", "smooth"])

    def test_ranking_can_follow_full_audit_change_and_size_band(self):
        first = rectangle("first", [[0, 0], [100, 0], [100, 50], [0, 50]])
        second = rectangle("second", [[0, 0], [120, 0], [120, 60], [0, 60]])
        payload = {"/capture/camera_0/image": {"0": [first], "1": [second]}}
        audit = [
            {"topic": "/capture/camera_0/image", "frame": 0,
             "shape_id": "first", "change_ratio": .01, "size_band": "small"},
            {"topic": "/capture/camera_0/image", "frame": 1,
             "shape_id": "second", "change_ratio": .02, "size_band": "large"},
        ]

        ranked = rank_candidates(
            payload, minimum_bbox_area=1000, limit=5,
            audit_results=audit, size_band="large")

        self.assertEqual([row["shape_id"] for row in ranked], ["second"])
        self.assertEqual(ranked[0]["_audit_result"]["change_ratio"], .02)

    def test_render_review_is_read_only_and_writes_bounded_evidence(self):
        shape = rectangle(
            "sam3-camera_0-0-pallet-1",
            [[10, 10], [110, 10], [110, 70], [10, 70]])
        hole = [[30, 30], [34, 30], [34, 34], [30, 34]]
        shape["parts"][0]["holes"] = [hole]
        shape["holes"] = [hole]
        payload = {"/capture/camera_0/image": {"0": [shape]}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = root / "candidates.json"
            candidates.write_text(json.dumps(payload), encoding="utf-8")
            before = candidates.read_bytes()
            camera = root / "images" / "camera_0"
            camera.mkdir(parents=True)
            Image.new("RGB", (160, 100), "gray").save(camera / "000.png")

            report = render_review(
                candidates, root / "images", root / "output",
                minimum_bbox_area=1000, limit=1)

            self.assertEqual(candidates.read_bytes(), before)
            self.assertEqual(report["selected_count"], 1)
            self.assertEqual(report["adjusted_count"], 1)
            self.assertGreater(report["maximum_change_fraction"], 0)
            self.assertTrue(report["rows"][0]["proposed"])
            self.assertEqual(
                report["rows"][0]["proposed_shape_id"],
                "sam3-camera_0-0-pallet-1")
            self.assertGreater(report["rows"][0]["changed_pixels"], 0)
            self.assertTrue(Path(report["contact_sheet"]).is_file())
            self.assertTrue(Path(report["report_path"]).is_file())
            self.assertTrue(report["candidate_only"])
            self.assertFalse(report["human_truth"])
            self.assertFalse(report["approved_for_training"])


if __name__ == "__main__":
    unittest.main()
