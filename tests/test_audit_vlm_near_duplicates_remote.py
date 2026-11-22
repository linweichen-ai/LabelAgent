import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.audit_vlm_near_duplicates_remote import manifest_entries, report


class VlmNearDuplicateRemoteTests(unittest.TestCase):
    def row(self, split, sample, sequence="000"):
        return {
            "split": split, "sample_id": sample,
            "image_path": ("/home/clw/datasets/sfl-cpd15y/"
                           "20260920_cs-heli_Dev_000001/"
                           f"{sequence}/bevannotation_data/cameras_640x512/"
                           f"camera_0/{sample}.png"),
            "image_sha256": hashlib.sha256(sample.encode()).hexdigest(),
        }

    def write_rows(self, path: Path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows),
                        encoding="utf-8")

    def test_manifest_entries_are_bounded_to_frozen_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {name: root / f"{name}.jsonl"
                     for name in ("train", "dev", "test24")}
            self.write_rows(paths["train"], [self.row("train", "a")])
            self.write_rows(paths["dev"], [self.row("dev", "b", "008")])
            self.write_rows(paths["test24"], [self.row("test24", "c", "010")])
            entries = manifest_entries(paths)
        self.assertEqual(len(entries), 3)
        self.assertEqual({entry["split"] for entry in entries},
                         {"train", "dev", "test24"})

    def test_outside_path_is_rejected_before_ssh(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            row = self.row("train", "a")
            row["image_path"] = "/tmp/a.png"
            self.write_rows(path, [row])
            with self.assertRaisesRegex(ValueError, "outside"):
                manifest_entries({"train": path})

    def test_report_preserves_review_required_hits_and_limits(self):
        entries = [self.row("train", "a"), self.row("dev", "b", "008")]
        hit = {"left": "a", "right": "b", "threshold_hit": True}
        value = report(entries, {
            "images": 2, "cross_split_pairs": 1,
            "threshold_hits": [hit], "nearest_pairs": [hit],
        }, "factory-4090")
        self.assertEqual(value["status"], "review_required")
        self.assertFalse(value["remote_writes"])
        self.assertFalse(value["pixels_downloaded"])


if __name__ == "__main__":
    unittest.main()
