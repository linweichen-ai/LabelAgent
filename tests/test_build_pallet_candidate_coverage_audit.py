import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from tools.build_pallet_candidate_coverage_audit import build


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PalletCandidateCoverageAuditTest(unittest.TestCase):
    def _fixture(self, root: Path):
        contact = root / "contact.jpg"
        Image.new("RGB", (20, 10), "gray").save(contact)
        localization_path = root / "localization.json"
        base_rows = []
        for camera in range(4):
            triggered = camera == 1
            base_rows.append({
                "sequence": "010", "camera": camera, "frame_id": "9",
                "target_obj_id": camera + 1, "localization_pass": True,
                "instance_isolation_pass": not triggered,
                "metrics": {"maximum_other_projected_box_fraction":
                            0.6 if triggered else 0.0},
            })
        localization = {
            "schema_version": "labelagent.pallet_box3d_roi_sam.v1",
            "production_deployed": False, "workbench_writes": 0,
            "image_count": 4, "contact_sheet": str(contact),
            "contact_sheet_sha256": digest(contact), "rows": base_rows,
        }
        localization_path.write_text(json.dumps(localization), encoding="utf-8")
        conflict_path = root / "conflict.json"
        conflict_path.write_text(json.dumps({
            "schema_version": "labelagent.pallet_box3d_overlap_pairs.v1",
            "production_deployed": False, "workbench_writes": 0,
            "source_report_sha256": digest(localization_path),
            "selected_source_row_count": 4, "skipped_no_conflict_count": 3,
            "rows": [{"sequence": "010", "camera": 1, "frame_id": "9"}],
        }), encoding="utf-8")
        route_path = root / "routes.json"
        route_path.write_text(json.dumps({
            "schema_version": "labelagent.pallet_visibility_candidate_routes.v1",
            "production_deployed": False, "workbench_writes": 0,
            "rows": [{"sequence": "010", "camera": 1, "frame_id": "9",
                      "route": "depth_split_candidate", "requires_review": True,
                      "write_human_truth": False, "write_training_data": False}],
        }), encoding="utf-8")
        return localization_path, conflict_path, route_path

    def test_builds_aligned_positive_and_negative_coverage(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); sources = self._fixture(root)
            report = build(*sources, root / "out")
            self.assertEqual(report["image_count"], 4)
            self.assertEqual(report["safe_skip_count"], 3)
            self.assertEqual(report["trigger_count"], 1)
            self.assertEqual(report["false_trigger_count"], 0)
            self.assertEqual(report["missed_trigger_count"], 0)
            self.assertEqual(report["candidate_count"], 1)
            self.assertTrue((root / "out" / "index.html").is_file())

    def test_rejects_route_on_negative_control(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); sources = self._fixture(root)
            routes = json.loads(sources[2].read_text(encoding="utf-8"))
            routes["rows"].append({
                "sequence": "010", "camera": 2, "frame_id": "9",
                "route": "depth_split_candidate", "requires_review": True,
                "write_human_truth": False, "write_training_data": False})
            sources[2].write_text(json.dumps(routes), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identities"):
                build(*sources, root / "out")


if __name__ == "__main__":
    unittest.main()
