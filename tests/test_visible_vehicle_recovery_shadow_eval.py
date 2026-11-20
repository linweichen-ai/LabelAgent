import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "evaluate_visible_vehicle_recovery_shadow",
    ROOT / "tools" / "evaluate_visible_vehicle_recovery_shadow.py")
shadow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(shadow)


def image_bytes(color):
    output = io.BytesIO()
    Image.new("RGB", (80, 60), color).save(output, "PNG")
    return output.getvalue()


class VisibleVehicleShadowEvaluationTests(unittest.TestCase):
    def test_load_samples_rejects_duplicate_scope(self):
        payload = {
            "schema_version": "labelagent.visible_vehicle_shadow_samples.v1",
            "samples": [
                {"topic": "/camera3", "frame": 1},
                {"topic": "/camera3", "frame": 1},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                shadow.load_samples(path)

    def test_contact_sheet_preserves_all_tiles(self):
        first = shadow.render_tile(image_bytes("red"), None, "camera0 f1")
        second = shadow.render_tile(
            image_bytes("blue"), image_bytes("green"), "camera3 f38")
        sheet = shadow.render_contact_sheet([first, second], columns=2)
        self.assertEqual(first.size, (640, 300))
        self.assertEqual(sheet.size, (1280, 300))

    def test_load_reference_session_selects_exact_media(self):
        with tempfile.TemporaryDirectory() as directory:
            media = Path(directory) / "sample.mcap"
            media.write_bytes(b"mcap")
            state = Path(directory) / "sessions.json"
            session = {"truth": {"3": [{"label": "other_robot"}]}}
            state.write_text(json.dumps({str(media.resolve()): session}),
                             encoding="utf-8")
            self.assertEqual(
                shadow.load_reference_session(state, media.resolve()), session)

    def test_reference_alignment_reports_same_label_polygon_iou(self):
        candidate = {
            "label": "other_robot",
            "points": [[0, 0], [9, 0], [9, 9], [0, 9]],
        }
        truth = [{
            "label": "other_robot",
            "points": [[0, 0], [9, 0], [9, 9], [0, 9]],
        }, {
            "label": "车辆",
            "points": [[15, 15], [19, 15], [19, 19], [15, 19]],
        }]
        result = shadow.reference_alignment(candidate, truth, (20, 20))
        self.assertEqual(result["target_count"], 2)
        self.assertEqual(result["same_label_target_count"], 1)
        self.assertEqual(result["best_same_label_iou"], 1.0)
        self.assertEqual(result["best_any_label"], "other_robot")

    def test_reference_alignment_keeps_absent_candidate_explicit(self):
        result = shadow.reference_alignment(None, [{
            "label": "other_robot",
            "points": [[0, 0], [4, 0], [4, 4]],
        }], (10, 10))
        self.assertFalse(result["candidate_present"])
        self.assertEqual(result["target_count"], 1)
        self.assertEqual(result["best_same_label_iou"], 0.0)

    def test_without_fast_route_does_not_mutate_source_config(self):
        source = {
            "llm_url": "http://primary/v1", "llm_model": "primary",
            "llm_fast_url": "http://fast/v1", "llm_fast_model": "fast",
            "llm_fast_timeout_seconds": 30,
        }
        primary = shadow.without_fast_route(source)
        self.assertEqual(primary, {
            "llm_url": "http://primary/v1", "llm_model": "primary",
        })
        self.assertEqual(source["llm_fast_model"], "fast")


if __name__ == "__main__":
    unittest.main()
