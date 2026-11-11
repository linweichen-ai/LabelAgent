import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from jsonschema import Draft202012Validator, FormatChecker

from tools.build_candidate_batch_manifest_v2 import build_manifest, canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/labelagent_candidate_batch_plan.json"
CONFIG_SHA = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
SAM_SHA = "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"
VLM_SHA = "5cd452860dc1e9c29dd71cc3cef7f39b338b7a40793f7a260655c2d3568f3661"


class CandidateBatchManifestV2Tests(unittest.TestCase):
    def test_static_batch_plan_is_no_execute_append_only_and_fail_closed(self):
        plan = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertFalse(plan["execution_enabled"])
        self.assertEqual(plan["write_policy"]["mode"],
                         "append_only_explicit_new_version")
        self.assertFalse(plan["write_policy"]["overwrite_human_truth"])
        self.assertTrue(plan["split_policy"]["test24"]["evaluation_only"])
        self.assertFalse(plan["split_policy"]["test24"][
            "training_return_allowed"])
        for trigger in (
            "new_label_or_unknown", "low_confidence", "vlm_sam_conflict"
        ):
            self.assertEqual(plan["review_routing"][trigger],
                             "human_review_required")
        self.assertFalse(plan["review_routing"]["automatic_truth_promotion"])

    def make_pack(self, root, split, prefix):
        directory = root / split
        directory.mkdir(parents=True)
        names = [
            f"{prefix}_1789897538282478905_camera_0.png",
            f"{prefix}_1789897538282478905_camera_1.png",
        ]
        for index, name in enumerate(names):
            (directory / name).write_bytes(f"{split}-image-{index}".encode())
        coco = {
            "images": [
                {"id": index + 1, "file_name": name, "width": 640, "height": 512}
                for index, name in enumerate(names)
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1},
                {"id": 2, "image_id": 2, "category_id": 2},
            ],
            "categories": [{"id": 1, "name": "pallet"}, {"id": 2, "name": "person"}],
        }
        (directory / "_preannotations.coco.json").write_text(
            json.dumps(coco), encoding="utf-8"
        )

    def build(self, root, split):
        return build_manifest(
            root, split, "sfl-cpd15y", "batch-a",
            "candidate-batch-test-v1", CONFIG_SHA, SAM_SHA, VLM_SHA,
        )

    def validate(self, manifest):
        schema = json.loads((
            ROOT / "schemas/candidate-batch-manifest-v2.schema.json"
        ).read_text(encoding="utf-8"))
        return list(Draft202012Validator(
            schema, format_checker=FormatChecker()).iter_errors(manifest))

    def test_train_and_test_are_separate_versioned_candidate_namespaces(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_pack(root, "train", "000")
            self.make_pack(root, "test", "010")
            train = self.build(root, "train")
            test = self.build(root, "test")
        self.assertEqual(self.validate(train), [])
        self.assertEqual(self.validate(test), [])
        self.assertEqual(train["split"], "train72")
        self.assertEqual(test["split"], "test24")
        self.assertFalse(train["split_policy"]["evaluation_only"])
        self.assertTrue(test["split_policy"]["evaluation_only"])
        self.assertTrue(all("/train72/" in item["output_candidate_path"]
                            for item in train["entries"]))
        self.assertTrue(all("/test24/" in item["output_candidate_path"]
                            for item in test["entries"]))
        self.assertTrue(set(item["idempotency_key"] for item in train["entries"])
                          .isdisjoint(item["idempotency_key"] for item in test["entries"]))

    def test_every_entry_binds_hashes_paths_and_fail_closed_review_routing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_pack(root, "train", "000")
            manifest = self.build(root, "train")
        for item in manifest["entries"]:
            self.assertEqual(item["preannotation_sha256"],
                             manifest["preannotation_source"]["sha256"])
            self.assertEqual(item["sam3_model_sha256"], SAM_SHA)
            self.assertEqual(item["vlm_config_sha256"], VLM_SHA)
            self.assertEqual(item["pipeline_config_sha256"], CONFIG_SHA)
            self.assertFalse(item["truth_eligible"])
            self.assertFalse(item["training_return_allowed"])
            self.assertFalse(item["automatic_truth_promotion"])
            self.assertEqual(item["output_write_mode"],
                             "append_only_explicit_new_version")
            self.assertNotIn("truth", item["output_candidate_path"])
            self.assertEqual(item["human_review_required_if"], [
                "new_label_or_unknown", "low_confidence", "vlm_sam_conflict"
            ])
        self.assertEqual(manifest["review_routing"]["low_confidence"],
                         "human_review_required")
        self.assertFalse(manifest["write_policy"]["overwrite_human_truth"])

    def test_manifest_is_byte_stable_and_content_hash_covers_versioned_body(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_pack(root, "test", "010")
            first = self.build(root, "test")
            second = self.build(root, "test")
        self.assertEqual(first, second)
        body = {key: value for key, value in first.items()
                if key != "manifest_content_sha256"}
        self.assertEqual(first["manifest_content_sha256"], canonical_sha256(body))
        self.assertEqual(first["gpu_calls"], 0)
        self.assertEqual(first["model_calls"], 0)

    def test_real_train72_and_test24_manifests_are_strictly_separated(self):
        train = json.loads((
            ROOT / "docs/test-results/candidate-batch-manifest-train72-20261111.json"
        ).read_text(encoding="utf-8"))
        test = json.loads((
            ROOT / "docs/test-results/candidate-batch-manifest-test24-20261111.json"
        ).read_text(encoding="utf-8"))
        self.assertEqual(self.validate(train), [])
        self.assertEqual(self.validate(test), [])
        self.assertEqual((train["entry_count"], test["entry_count"]), (72, 24))
        self.assertEqual({item["sequence"] for item in train["entries"]},
                         {f"{index:03d}" for index in range(9)})
        self.assertEqual({item["sequence"] for item in test["entries"]},
                         {"010", "011", "012"})
        self.assertTrue(all(item["split"] == "train72"
                            for item in train["entries"]))
        self.assertTrue(all(item["split"] == "test24"
                            for item in test["entries"]))
        self.assertTrue(test["split_policy"]["evaluation_only"])
        self.assertFalse(test["split_policy"]["test24_training_return_allowed"])
        self.assertTrue(all(item["evaluation_only"] for item in test["entries"]))
        self.assertTrue(all(not item["training_return_allowed"]
                            for item in test["entries"]))
        train_inputs = {item["input_id"] for item in train["entries"]}
        test_inputs = {item["input_id"] for item in test["entries"]}
        self.assertTrue(train_inputs.isdisjoint(test_inputs))
        train_outputs = {item["output_candidate_path"] for item in train["entries"]}
        test_outputs = {item["output_candidate_path"] for item in test["entries"]}
        self.assertTrue(train_outputs.isdisjoint(test_outputs))

    def test_real_manifests_bind_current_config_models_and_preannotations(self):
        readback = json.loads((
            ROOT / "docs/test-results/sam3-96-pack-remote-readback-20261111.json"
        ).read_text(encoding="utf-8"))
        for name, split_key in (
            ("candidate-batch-manifest-train72-20261111.json", "train72"),
            ("candidate-batch-manifest-test24-20261111.json", "test24"),
        ):
            manifest = json.loads((ROOT / "docs/test-results" / name)
                                  .read_text(encoding="utf-8"))
            self.assertEqual(manifest["pipeline_config_sha256"], CONFIG_SHA)
            self.assertEqual(manifest["model_bindings"]["sam3_model_sha256"],
                             SAM_SHA)
            self.assertEqual(manifest["model_bindings"]["vlm_config_sha256"],
                             VLM_SHA)
            self.assertEqual(
                manifest["preannotation_source"]["sha256"],
                readback["splits"][split_key]["preannotations_sha256"],
            )
            body = {key: value for key, value in manifest.items()
                    if key != "manifest_content_sha256"}
            self.assertEqual(manifest["manifest_content_sha256"],
                             canonical_sha256(body))
            self.assertEqual(manifest["gpu_calls"], 0)
            self.assertEqual(manifest["model_calls"], 0)
            self.assertFalse(manifest["write_policy"]["overwrite_human_truth"])
            self.assertTrue(all("truth" not in item["output_candidate_path"]
                                for item in manifest["entries"]))

    def test_static_validation_receipt_schema_and_file_hashes(self):
        receipt = json.loads((
            ROOT / "docs/test-results/candidate-batch-static-validation-receipt-20261111.json"
        ).read_text(encoding="utf-8"))
        schema = json.loads((
            ROOT / "schemas/candidate-batch-static-validation-receipt.schema.json"
        ).read_text(encoding="utf-8"))
        errors = list(Draft202012Validator(
            schema, format_checker=FormatChecker()).iter_errors(receipt))
        self.assertEqual(errors, [], [error.message for error in errors])
        for artifact in [receipt["tool"], receipt["config"]]:
            self.assertEqual(hashlib.sha256(
                (ROOT / artifact["path"]).read_bytes()).hexdigest(),
                artifact["sha256"])
        for item in receipt["manifests"]:
            path = ROOT / item["path"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                             item["file_sha256"])
            manifest = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["manifest_content_sha256"],
                             item["manifest_content_sha256"])
        self.assertEqual(receipt["execution"]["gpu_calls"], 0)
        self.assertFalse(receipt["execution"]["human_truth_modified"])


if __name__ == "__main__":
    unittest.main()
