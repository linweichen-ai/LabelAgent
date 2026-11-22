import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tools.build_ego_tail_focused_review_pack import write_contact_pages


class EgoTailFocusedReviewPackTests(unittest.TestCase):
    def test_writes_contact_page_with_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (768, 768), (20, 30, 40)).save(
                root / "c1-focused.png")
            pages = write_contact_pages(root, [{
                "component_id": "c1",
                "camera_rank": 73,
                "policy_queue": "deferred_manual_review",
                "focused_image": "c1-focused.png",
            }])
            self.assertEqual(len(pages), 1)
            self.assertEqual(pages[0]["first_component_index"], 1)
            self.assertEqual(pages[0]["last_component_index"], 1)
            self.assertEqual(len(pages[0]["sha256"]), 64)
            self.assertTrue((root / pages[0]["image"]).is_file())

    def test_rejects_non_positive_page_size(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "positive"):
                write_contact_pages(Path(directory), [], page_size=0)


if __name__ == "__main__":
    unittest.main()
