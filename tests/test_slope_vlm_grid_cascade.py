import io
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from tools.evaluate_slope_vlm_grid_cascade import (
    DUAL_IMAGE_INSTRUCTION, GRID_INSTRUCTION, cell_center, grid_overlay,
    json_sha256, load_completed_row, validate_decision,
)


class SlopeVlmGridCascadeTests(unittest.TestCase):
    def test_cell_mapping_and_grid_render(self):
        self.assertEqual(cell_center("A1"), (40, 32))
        self.assertEqual(cell_center("H8"), (600, 480))
        source = io.BytesIO()
        Image.new("RGB", (640, 512), "black").save(source, "PNG")
        marked = Image.open(io.BytesIO(grid_overlay(source.getvalue())))
        self.assertEqual(marked.size, (640, 512))

    def test_prompt_has_no_color_rule(self):
        for prompt in (GRID_INSTRUCTION, DUAL_IMAGE_INSTRUCTION):
            self.assertNotIn("yellow", prompt.lower())
            self.assertNotIn("orange", prompt.lower())
        self.assertIn("Image 1", DUAL_IMAGE_INSTRUCTION)
        self.assertIn("Image 2", DUAL_IMAGE_INSTRUCTION)

    def test_presence_cell_contract(self):
        value = validate_decision({"slope_visible": True, "cell_id": "H7",
                                   "confidence": 0.8,
                                   "reason": "central cross is on incline"})
        self.assertEqual(value["cell_id"], "H7")
        with self.assertRaises(ValueError):
            validate_decision({"slope_visible": False, "cell_id": "H7",
                               "confidence": 0.8, "reason": "no ramp"})

    def test_completed_row_resume_is_hash_bound(self):
        source = {"sample_id": "sample", "image_sha256": "a" * 64}
        bindings = {"state_sha256": "b" * 64}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "grid.jpg").write_bytes(b"grid")
            (root / "overlay.jpg").write_bytes(b"overlay")
            row = {"status": "succeeded", "sample_id": "sample",
                   "image_sha256": "a" * 64, "bindings": bindings,
                   "grid_image": "grid.jpg", "overlay": "overlay.jpg"}
            path = root / "row.json"
            path.write_text(json.dumps(row), encoding="utf-8")
            self.assertEqual(load_completed_row(path, source, bindings, root), row)
            row["image_sha256"] = "c" * 64
            path.write_text(json.dumps(row), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_completed_row(path, source, bindings, root)

    def test_schema_hash_is_order_stable(self):
        self.assertEqual(json_sha256({"a": 1, "b": 2}),
                         json_sha256({"b": 2, "a": 1}))


if __name__ == "__main__":
    unittest.main()
