import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tools import build_mixed_hard_review_pack as packet


class FakeVideo:
    def __init__(self, _path):
        pass

    def frame(self, _topic, index):
        from io import BytesIO
        buffer = BytesIO()
        Image.new("RGB", (80, 60), (index + 20, 30, 40)).save(buffer, "PNG")
        return buffer.getvalue()


class MixedHardReviewPackTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def image(self, path, color):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (80, 60), color).save(path)

    def sources(self):
        dev = self.root / "dev"
        decisions = []
        for index in range(8):
            sample = f"dev-{index}"
            self.image(dev / f"{sample}.png", (150 + index, 20, 30))
            decisions.append({"sample_id": sample,
                              "assistant_visible_slope_candidate": index < 4,
                              "view": "bounded assistant candidate"})
        dev_review = self.root / "dev-review.json"
        dev_review.write_text(json.dumps({
            "split": "dev8", "human_truth": False, "training_authorized": False,
            "decisions": decisions}), encoding="utf-8")
        mcap = self.root / "source.mcap"
        mcap.write_bytes(b"bounded-mcap-fixture")
        slope = self.root / "slope.json"
        slope.write_text(json.dumps({
            "mcap": str(mcap.resolve()), "topic": "/camera", "mode": "candidate",
            "rows": [{"frame": index, "candidate_count": 0, "filtered_count": 0}
                     for index in range(24)]}), encoding="utf-8")
        pallet_root = self.root / "pallet-images"
        frames = []
        for index in range(2):
            frame_id = str(100 + index)
            self.image(pallet_root / "006" / "camera_1" / f"{frame_id}.png",
                       (10, 160 + index, 30))
            frames.append({"sequence": "006", "frame_id": frame_id, "camera": 1,
                           "pallet_label_count": 9, "review_scope_obj_ids": ["1", "2"]})
        pallet = self.root / "pallet.json"
        pallet.write_text(json.dumps({
            "candidate_only": True, "human_pixel_truth": False,
            "approved_for_training": False, "frames": frames}), encoding="utf-8")
        test24 = self.root / "test24.jsonl"
        test24.write_text(json.dumps({
            "split": "test24", "evaluation_only": True,
            "image_sha256": "f" * 64}) + "\n", encoding="utf-8")
        return dev, dev_review, slope, mcap, pallet, pallet_root, test24

    def test_builds_34_unique_candidate_only_frames(self):
        sources = self.sources()
        output = self.root / "output"
        manifest = packet.build(
            dev_dir=sources[0], dev_review=sources[1], slope_results=sources[2],
            slope_mcap=sources[3], pallet_manifest=sources[4],
            pallet_image_root=sources[5], test24_jsonl=sources[6], output=output,
            video_factory=FakeVideo)
        self.assertEqual(manifest["summary"]["frames"], 34)
        self.assertEqual(manifest["summary"]["unique_source_hashes"], 34)
        self.assertEqual(manifest["summary"]["test24_intersection_count"], 0)
        self.assertTrue(manifest["candidate_only"])
        self.assertFalse(manifest["human_truth"])
        self.assertFalse(manifest["approved_for_training"])
        self.assertEqual(len(list((output / "raw").glob("*.jpg"))), 34)
        self.assertTrue((output / "contact-sheet.jpg").is_file())
        self.assertTrue((output / "index.html").is_file())
        self.assertTrue((output / "receipt.json").is_file())

    def test_rejects_test24_intersection(self):
        sources = self.sources()
        first = next(sources[0].glob("*.png"))
        test_hash = packet.file_sha256(first)
        sources[6].write_text(json.dumps({
            "split": "test24", "evaluation_only": True,
            "image_sha256": test_hash}) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "test24 image intersection"):
            packet.build(
                dev_dir=sources[0], dev_review=sources[1], slope_results=sources[2],
                slope_mcap=sources[3], pallet_manifest=sources[4],
                pallet_image_root=sources[5], test24_jsonl=sources[6],
                output=self.root / "blocked", video_factory=FakeVideo)


if __name__ == "__main__":
    unittest.main()
