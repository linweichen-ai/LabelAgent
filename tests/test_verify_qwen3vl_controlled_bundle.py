import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from tools.build_qwen3vl_controlled_bundle import build
from tools.verify_qwen3vl_controlled_bundle import verify


class ControlledBundleVerificationTests(unittest.TestCase):
    def make_bundle(self, root: Path) -> Path:
        adapter = root / "training/adapter"
        adapter.mkdir(parents=True)
        values = {
            "adapter_model.safetensors": b"weights",
            "adapter_config.json": json.dumps({
                "base_model_name_or_path": "/models/Qwen3-VL-2B-Instruct",
            }).encode(),
        }
        rows = []
        for name, payload in values.items():
            (adapter / name).write_bytes(payload)
            rows.append({"path": f"training/adapter/{name}", "bytes": len(payload),
                         "sha256": hashlib.sha256(payload).hexdigest()})
        manifest = root / "release.json"
        manifest.write_text(json.dumps({
            "status": "passed_local_release_audit",
            "candidate": {"adapter_sha256": hashlib.sha256(b"weights").hexdigest()},
            "boundaries": {"weights_published": False,
                           "distribution_approved": False,
                           "human_truth": False, "production_deployed": False,
                           "test24_used_for_training": False},
            "adapter_files": rows,
        }))
        bundle = root / "bundle.zip"
        build(root, manifest, bundle)
        return bundle

    def test_independent_hash_and_member_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = self.make_bundle(Path(directory))
            expected = hashlib.sha256(bundle.read_bytes()).hexdigest()
            value = verify(bundle, expected)
        self.assertEqual(value["status"], "passed")
        self.assertFalse(value["tensor_scan"]["performed"])

    def test_rejects_wrong_bundle_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = self.make_bundle(Path(directory))
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                verify(bundle, "0" * 64)

    def test_rejects_unmanifested_zip_member(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = self.make_bundle(Path(directory))
            with zipfile.ZipFile(bundle, "a") as archive:
                archive.writestr("extra.txt", b"unexpected")
            with self.assertRaisesRegex(ValueError, "differ"):
                verify(bundle)


if __name__ == "__main__":
    unittest.main()
