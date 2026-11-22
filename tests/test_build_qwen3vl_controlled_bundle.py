import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from tools.build_qwen3vl_controlled_bundle import build


class ControlledBundleTests(unittest.TestCase):
    def fixture(self, root: Path):
        adapter = root / "training/adapter"
        adapter.mkdir(parents=True)
        values = {
            "adapter_model.safetensors": b"weights",
            "adapter_config.json": json.dumps({
                "base_model_name_or_path": "/models/Qwen3-VL-2B-Instruct",
                "r": 8,
            }).encode(),
        }
        rows = []
        for name, payload in values.items():
            (adapter / name).write_bytes(payload)
            rows.append({"path": f"training/adapter/{name}", "bytes": len(payload),
                         "sha256": hashlib.sha256(payload).hexdigest()})
        manifest = {
            "status": "passed_local_release_audit",
            "candidate": {"adapter_sha256": hashlib.sha256(b"weights").hexdigest()},
            "boundaries": {"weights_published": False,
                           "distribution_approved": False},
            "adapter_files": rows,
        }
        path = root / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def test_builds_deterministic_safe_local_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.fixture(root)
            one, two = root / "one.zip", root / "two.zip"
            first = build(root, manifest, one)
            second = build(root, manifest, two)
            self.assertEqual(first["bundle_sha256"], second["bundle_sha256"])
            self.assertFalse(first["remote_uploaded"])
            with zipfile.ZipFile(one) as archive:
                self.assertIn("bundle-manifest.json", archive.namelist())
                self.assertIn("adapter/adapter_model.safetensors", archive.namelist())

    def test_rejects_adapter_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.fixture(root)
            (root / "training/adapter/adapter_model.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "drift"):
                build(root, manifest, root / "bad.zip")

    def test_rejects_public_distribution_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.fixture(root)
            value = json.loads(manifest.read_text())
            value["boundaries"]["distribution_approved"] = True
            manifest.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "local-only"):
                build(root, manifest, root / "bad.zip")


if __name__ == "__main__":
    unittest.main()
