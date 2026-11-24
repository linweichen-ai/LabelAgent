import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tools import build_mixed_hard_adjudication_workspace as adjudication
from tools import diversify_mixed_hard_review_pack as diversify


class DiverseMixedHardReviewPackTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def image(self, path, color):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (80, 60), color).save(path)

    def packet(self):
        packet = self.root / "parent"
        (packet / "raw").mkdir(parents=True)
        (packet / "review").mkdir()
        rows = []
        for index in range(24):
            sample_id = f"parent-{index}"
            raw = packet / "raw" / f"{sample_id}.jpg"
            review = packet / "review" / f"{sample_id}.jpg"
            self.image(raw, ((index * 10 + 10) % 256, 20, 30))
            self.image(review, ((index * 10 + 10) % 256, 80, 110))
            rows.append({
                "sample_id": sample_id,
                "cohort": "slope_hard_negative_candidate",
                "source_image_sha256": diversify.file_sha256(raw),
                "raw_image": raw.relative_to(packet).as_posix(),
                "review_image": review.relative_to(packet).as_posix(),
                "review_note": "candidate",
                "source_metadata": {"frame": index},
                "review_status": "pending_human_review",
                "candidate_only": True,
                "human_truth": False,
                "approved_for_training": False,
            })
        manifest = {
            "schema_version": "labelagent.mixed-hard-review-pack.v1",
            "summary": {"frames": 24, "test24_intersection_count": 0,
                        "human_truth_frames": 0, "training_approved_frames": 0},
            "rows": rows,
            "candidate_only": True,
            "human_truth": False,
            "approved_for_training": False,
            "training_authorized": False,
            "test24_used": False,
            "production_deployed": False,
        }
        manifest_path = packet / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        files = [manifest_path, *sorted((packet / "raw").glob("*")),
                 *sorted((packet / "review").glob("*"))]
        receipt = {
            "schema_version": "labelagent.mixed-hard-review-pack-receipt.v1",
            "manifest_sha256": diversify.file_sha256(manifest_path),
            "files": [{"path": p.relative_to(packet).as_posix(),
                       "sha256": diversify.file_sha256(p), "bytes": p.stat().st_size}
                      for p in files],
        }
        (packet / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
        return packet, rows

    def inputs(self):
        packet, rows = self.packet()
        test24 = self.root / "test24.jsonl"
        test24.write_text(json.dumps({"split": "test24", "evaluation_only": True,
                                      "image_sha256": "f" * 64}) + "\n",
                          encoding="utf-8")
        samples = []
        for index in range(4):
            source = self.root / "pool" / f"new-{index}.png"
            self.image(source, (180, index * 25 + 30, 40))
            samples.append({
                "sample_id": f"new-{index}",
                "source_path": str(source),
                "source_image_sha256": diversify.file_sha256(source),
                "cohort": "batchb_flat_negative_candidate" if index < 2
                          else "batchb_ramp_candidate",
                "review_note": "declared candidate",
                "source_metadata": {"dataset": "batch-b", "sequence": "003",
                                    "camera": index % 2, "frame_index": index},
            })
        selection = self.root / "selection.json"
        selection.write_text(json.dumps({
            "schema_version": diversify.SELECTION_SCHEMA,
            "replace_sample_ids": [row["sample_id"] for row in rows[:4]],
            "samples": samples,
            "test24_jsonl_sha256": diversify.file_sha256(test24),
            "candidate_only": True,
            "human_truth": False,
            "approved_for_training": False,
            "test24_used": False,
        }), encoding="utf-8")
        return packet, selection, test24, samples

    def test_builds_hash_bound_diverse_candidate_packet(self):
        packet, selection, test24, _ = self.inputs()
        output = self.root / "output"
        manifest = diversify.build(parent_packet=packet,
                                    selection_manifest=selection,
                                    test24_jsonl=test24, output=output)
        self.assertEqual(manifest["summary"]["frames"], 24)
        self.assertEqual(manifest["summary"]["replaced_parent_frames"], 4)
        self.assertEqual(manifest["summary"]["added_diverse_frames"], 4)
        self.assertEqual(manifest["summary"]["test24_intersection_count"], 0)
        self.assertTrue(manifest["candidate_only"])
        self.assertFalse(manifest["human_truth"])
        self.assertFalse(manifest["approved_for_training"])
        self.assertFalse(manifest["test24_used"])
        self.assertEqual(manifest["external_api_calls"], 0)
        self.assertTrue((output / "contact-sheet.jpg").is_file())
        self.assertTrue((output / "index.html").is_file())
        self.assertTrue((output / "receipt.json").is_file())
        checked, _ = adjudication.validate_packet(output)
        self.assertEqual(checked["summary"]["frames"], 24)

    def test_rejects_test24_intersection(self):
        packet, selection, test24, samples = self.inputs()
        test24.write_text(json.dumps({"split": "test24", "evaluation_only": True,
                                      "image_sha256": samples[0]["source_image_sha256"]}) + "\n",
                          encoding="utf-8")
        payload = json.loads(selection.read_text(encoding="utf-8"))
        payload["test24_jsonl_sha256"] = diversify.file_sha256(test24)
        selection.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "test24 image intersection"):
            diversify.build(parent_packet=packet, selection_manifest=selection,
                              test24_jsonl=test24, output=self.root / "blocked")

    def test_rejects_source_hash_drift(self):
        packet, selection, test24, samples = self.inputs()
        self.image(Path(samples[0]["source_path"]), (1, 2, 3))
        with self.assertRaisesRegex(ValueError, "new sample hash differs"):
            diversify.build(parent_packet=packet, selection_manifest=selection,
                              test24_jsonl=test24, output=self.root / "blocked")

    def test_rejects_unsafe_training_boundary(self):
        packet, selection, test24, _ = self.inputs()
        payload = json.loads(selection.read_text(encoding="utf-8"))
        payload["approved_for_training"] = True
        selection.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unsafe selection manifest boundary"):
            diversify.build(parent_packet=packet, selection_manifest=selection,
                              test24_jsonl=test24, output=self.root / "blocked")


if __name__ == "__main__":
    unittest.main()
