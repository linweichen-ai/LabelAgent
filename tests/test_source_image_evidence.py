import hashlib
import io
import unittest

from PIL import Image, ImageDraw
from tools.source_image_evidence import guard_category, inspect_source


def encode(image, **kwargs):
    buffer = io.BytesIO()
    image.save(buffer, format=kwargs.pop("format", "PNG"), **kwargs)
    return buffer.getvalue()


class SourceEvidenceChecks(unittest.TestCase):
    def test_constant_colors_abstain_symmetrically(self):
        for color in ((127, 127, 127), (255, 255, 255), (255, 0, 0)):
            evidence = inspect_source(encode(Image.new("RGB", (19, 13), color)))
            self.assertFalse(evidence.available)
            self.assertEqual(evidence.channel_ranges, (0, 0, 0))
            for category in ("pallet", "person", "uncertain"):
                self.assertEqual(guard_category(category, evidence)["category"], "uncertain")

    def test_one_pixel_one_level_is_not_a_texture_threshold(self):
        image = Image.new("RGB", (19, 13), (127, 127, 127))
        image.putpixel((18, 12), (128, 127, 127))
        evidence = inspect_source(encode(image))
        self.assertTrue(evidence.available)
        self.assertEqual(evidence.channel_ranges, (1, 0, 0))
        for category in ("pallet", "person", "uncertain"):
            self.assertEqual(guard_category(category, evidence)["category"], category)

    def test_unmarked_source_needed_before_artificial_outline(self):
        source = Image.new("RGB", (19, 13), "white")
        raw = encode(source)
        ImageDraw.Draw(source).rectangle((3, 3, 10, 10), outline="magenta")
        self.assertFalse(inspect_source(raw).available)
        self.assertTrue(inspect_source(encode(source)).available)

    def test_corrupt_alpha_and_palette_abstain_explicitly(self):
        self.assertEqual(inspect_source(b"not an image").reason, "unreadable_source")
        for mode in ("RGBA", "P"):
            evidence = inspect_source(encode(Image.new(mode, (10, 10))))
            self.assertEqual(evidence.reason, "unsupported_mode")
            self.assertEqual(guard_category("person", evidence)["category"], "uncertain")

    def test_multiframe_abstains(self):
        a, b = Image.new("RGB", (10, 10), "red"), Image.new("RGB", (10, 10), "blue")
        evidence = inspect_source(encode(a, format="TIFF", save_all=True, append_images=[b]))
        self.assertEqual(evidence.reason, "unsupported_multiframe")

    def test_grayscale_and_jpeg_use_decoded_spatial_values(self):
        for mode in ("L", "RGB"):
            image = Image.new(mode, (19, 13), 127)
            evidence = inspect_source(encode(image, format="JPEG"))
            self.assertFalse(evidence.available)
            self.assertEqual(evidence.channel_ranges, (0, 0, 0))

    def test_raw_input_and_proposal_remain_unchanged_with_bound_trace(self):
        raw = encode(Image.new("RGB", (19, 13), "white"))
        proposal = {"category": "pallet", "reason": "retained raw model answer"}
        before = dict(proposal)
        decision = guard_category(proposal["category"], inspect_source(raw))
        self.assertEqual(proposal, before)
        self.assertEqual(decision["raw_category"], "pallet")
        self.assertEqual(decision["source_evidence"]["source_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(raw, encode(Image.new("RGB", (19, 13), "white")))

    def test_invalid_inputs_are_not_confident_predictions(self):
        with self.assertRaises(TypeError):
            inspect_source("file path is not image bytes")
        with self.assertRaises(ValueError):
            guard_category("other", inspect_source(b""))
        with self.assertRaises(TypeError):
            guard_category("pallet", {"available": True})


if __name__ == "__main__":
    unittest.main()
