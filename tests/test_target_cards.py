import hashlib
import unittest

from PIL import Image
from tools.evaluate_target_cards import card_payload, compose_target_views, fit_panel, pair_rows


class TargetCardChecks(unittest.TestCase):
    def test_entire_thin_source_fits_without_cropping(self):
        image = Image.new("RGB", (100, 10), "red")
        image.putpixel((0, 0), (0, 255, 0))
        image.putpixel((99, 9), (0, 0, 255))
        result, meta = fit_panel(image, 100, 100)
        self.assertEqual(meta["display_size"], [100, 10])
        self.assertEqual(result.getpixel((0, 45)), (0, 255, 0))
        self.assertEqual(result.getpixel((99, 54)), (0, 0, 255))

    def test_renderer_does_not_mutate_inputs_or_render_class_labels(self):
        views = [Image.new("RGB", (64, 32), color) for color in ("red", "green", "blue")]
        before = [v.tobytes() for v in views]
        cards, meta = compose_target_views(*views)
        self.assertEqual(before, [v.tobytes() for v in views])
        self.assertEqual(cards["overview"].size, (512, 512))
        self.assertEqual(cards["linked"].size, (1536, 512))
        self.assertFalse(meta["class_or_ID_text_rendered"])
        self.assertEqual(meta["neutral_panel_titles"], ["SELECTED", "SOURCE", "OUTLINE"])

    def test_payload_hides_case_roles_and_binds_image_budget(self):
        raw = b"synthetic image bytes"
        ref = {"path": "a.jpg", "sha256": hashlib.sha256(raw).hexdigest()}
        f = {"model": "labelagent-vlm", "categories": ["pallet", "person", "uncertain"],
             "temperature": 0, "max_tokens": 192, "prompts": {"overview": "neutral prompt"},
             "image_token_budgets": {"overview": 256}}
        c = {"id": "private_id", "expected_category": "person", "views": {"overview": [ref]}}
        a, ha = card_payload(f, {"a.jpg": (raw, "image/jpeg")}, c, "overview")
        b, hb = card_payload(f, {"a.jpg": (raw, "image/jpeg")}, {**c, "id": "changed", "expected_category": "pallet"}, "overview")
        self.assertEqual(a, b); self.assertEqual(ha, hb)
        _, hc = card_payload({**f, "image_token_budgets": {"overview": 768}}, {"a.jpg": (raw, "image/jpeg")}, c, "overview")
        self.assertNotEqual(ha, hc)

    def test_disagreement_and_invalid_pair_cannot_be_confirmed(self):
        cases = [{"id": "a", "group": "human"}, {"id": "b", "group": "human"}]
        rows = [{"case_id": "a", "view": "overview", "parsed": {"category": "pallet"}},
                {"case_id": "a", "view": "linked", "parsed": {"category": "person"}},
                {"case_id": "b", "view": "overview", "parsed": {"category": "pallet"}}]
        pairs = pair_rows(cases, rows)
        self.assertEqual(pairs[0]["consensus"], "uncertain")
        self.assertFalse(pairs[1]["valid_pair"])
        self.assertEqual(pairs[1]["consensus"], "uncertain")


if __name__ == "__main__":
    unittest.main()
