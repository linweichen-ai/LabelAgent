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


if __name__ == "__main__":
    unittest.main()
