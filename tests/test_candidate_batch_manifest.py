import json
from pathlib import Path
import tempfile
import unittest

from jsonschema import Draft202012Validator, FormatChecker

from tools.build_candidate_batch_manifest import build_manifest, canonical_sha256


ROOT = Path(__file__).resolve().parents[1]


class CandidateBatchManifestTests(unittest.TestCase):
    def make_pack(self, root):
        split = root / "train"
        split.mkdir(parents=True)
        names = [
            "000_1789897506782450197_camera_0.png",
            "000_1789897538282478905_camera_0.png",
            "001_1789897600000000000_camera_1.png",
        ]
        for index, name in enumerate(names):
            (split / name).write_bytes(f"image-{index}".encode())
        coco = {
            "images": [
                {"id": index + 1, "file_name": name, "width": 640, "height": 512}
                for index, name in enumerate(names)
            ],
            "annotations": [
                {"id": 1, "image_id": 2, "category_id": 1},
                {"id": 2, "image_id": 2, "category_id": 2},
                {"id": 3, "image_id": 3, "category_id": 1},
            ],
            "categories": [{"id": 1, "name": "pallet"}, {"id": 2, "name": "person"}],
        }
        (split / "_preannotations.coco.json").write_text(
            json.dumps(coco), encoding="utf-8"
        )

    def test_manifest_is_deterministic_candidate_only_and_excludes_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_pack(root)
            first = build_manifest(root, "train", "sfl-cpd15y", "batch-a")
            second = build_manifest(root, "train", "sfl-cpd15y", "batch-a")
        self.assertEqual(first, second)
        self.assertEqual(first["entry_count"], 2)
        self.assertEqual(first["fixed_holdout_exclusion"]["excluded_matches"], [
            "000_1789897506782450197_camera_0.png"
        ])
        self.assertFalse(first["preannotation_source"]["truth_eligible"])
        self.assertTrue(all(item["candidate_only"] for item in first["entries"]))
        self.assertTrue(all(not item["truth_eligible"] for item in first["entries"]))
        self.assertTrue(all(not item["approved_for_training"] for item in first["entries"]))
        body = {key: value for key, value in first.items()
                if key != "manifest_content_sha256"}
        self.assertEqual(first["manifest_content_sha256"], canonical_sha256(body))

    def test_limit_is_deterministic_and_does_not_reintroduce_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_pack(root)
            result = build_manifest(
                root, "train", "sfl-cpd15y", "batch-a", limit=1
            )
        self.assertEqual(result["entry_count"], 1)
        self.assertNotEqual(result["entries"][0]["frame_id"],
                            "1789897506782450197")

    def test_real_two_entry_smoke_manifest_validates_and_tool_hash_matches(self):
        receipt = json.loads((
            ROOT / "docs/test-results/candidate-batch-manifest-smoke-20261111.json"
        ).read_text(encoding="utf-8"))
        schema = json.loads((
            ROOT / "schemas/candidate-batch-manifest.schema.json"
        ).read_text(encoding="utf-8"))
        errors = list(Draft202012Validator(
            schema, format_checker=FormatChecker()
        ).iter_errors(receipt["manifest"]))
        self.assertEqual(errors, [], [error.message for error in errors])
        self.assertEqual(receipt["manifest"]["entry_count"],
                         len(receipt["manifest"]["entries"]))
        tool_hash = __import__("hashlib").sha256(
            (ROOT / receipt["tool"]["path"]).read_bytes()
        ).hexdigest()
        self.assertEqual(tool_hash, receipt["tool"]["sha256"])
        self.assertEqual(receipt["scope"]["gpu_calls"], 0)
        self.assertFalse(receipt["scope"]["remote_writes"])


if __name__ == "__main__":
    unittest.main()
