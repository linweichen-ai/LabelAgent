import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILDER = load("build_model_asset_manifest",
               ROOT / "tools/build_model_asset_manifest.py")
SERVER = load("local_vlm_server_for_manifest_test",
              ROOT / "tools/local_vlm_server.py")


class ModelAssetManifestTests(unittest.TestCase):
    def create_model(self, root):
        model = Path(root) / "model"
        model.mkdir()
        for name in sorted(BUILDER.REQUIRED_ASSETS):
            (model / name).write_text("{}\n", encoding="utf-8")
        (model / "generation_config.json").write_text(
            "{}\n", encoding="utf-8")
        (model / "merges.txt").write_text("a b\n", encoding="utf-8")
        (model / "model-00001-of-00001.safetensors").write_bytes(
            b"weights")
        (model / "README.md").write_text(
            "not a runtime asset\n", encoding="utf-8")
        return model

    def test_generated_manifest_round_trips_through_runtime_validator(self):
        with tempfile.TemporaryDirectory() as directory:
            model = self.create_model(directory)
            output = Path(directory) / "manifests" / "assets.json"
            receipt = BUILDER.write_manifest(model, output)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(receipt["files"], 8)
            self.assertNotIn("README.md",
                             {item["name"] for item in payload["files"]})
            self.assertEqual(len(SERVER.validate_model_asset_manifest(
                model, output)), 64)

    def test_output_inside_model_and_implicit_overwrite_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            model = self.create_model(directory)
            with self.assertRaisesRegex(ValueError, "outside"):
                BUILDER.write_manifest(model, model / "assets.json")
            output = Path(directory) / "assets.json"
            BUILDER.write_manifest(model, output)
            with self.assertRaises(FileExistsError):
                BUILDER.write_manifest(model, output)

    def test_changed_asset_invalidates_manifest_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            model = self.create_model(directory)
            output = Path(directory) / "assets.json"
            BUILDER.write_manifest(model, output)
            (model / "merges.txt").write_text(
                "changed tokenizer data\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "metadata differs"):
                SERVER.validate_model_asset_manifest(model, output)


if __name__ == "__main__":
    unittest.main()
