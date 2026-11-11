import json
from pathlib import Path
import unittest

from jsonschema import Draft202012Validator, FormatChecker


ROOT = Path(__file__).resolve().parents[1]


class Sam396PackRemoteReadbackTests(unittest.TestCase):
    def validator(self, name):
        schema = json.loads((ROOT / "schemas" / name).read_text(encoding="utf-8"))
        return Draft202012Validator(schema, format_checker=FormatChecker())

    def test_new_remote_readback_is_schema_valid_and_truth_blocked(self):
        receipt = json.loads((
            ROOT / "docs/test-results/sam3-96-pack-remote-readback-20261111.json"
        ).read_text(encoding="utf-8"))
        errors = list(self.validator(
            "sam3-96-pack-remote-readback.schema.json").iter_errors(receipt))
        self.assertEqual(errors, [], [error.message for error in errors])
        self.assertEqual(receipt["splits"]["train72"]["image_count"], 72)
        self.assertEqual(receipt["splits"]["test24"]["image_count"], 24)
        self.assertFalse(receipt["splits"]["train72"]["human_annotations_present"])
        self.assertFalse(receipt["splits"]["test24"]["human_annotations_present"])
        self.assertFalse(receipt["role_boundary"]["formal_training_authorized"])

    def test_current_manifest_points_to_new_readback_and_stays_blocked(self):
        manifest = json.loads((
            ROOT / "configs/sam3_sfl_cpd15y_peft_data_manifest.blocked.json"
        ).read_text(encoding="utf-8"))
        errors = list(self.validator(
            "sam3-peft-data-manifest.schema.json").iter_errors(manifest))
        self.assertEqual(errors, [], [error.message for error in errors])
        self.assertTrue(manifest["inventory"]["remote_pack_present"])
        self.assertTrue(manifest["inventory"]["machine_preannotations_only"])
        self.assertFalse(manifest["truth"]["train_annotations_verified"])
        self.assertFalse(manifest["authorization"]["formal_training_authorized"])


if __name__ == "__main__":
    unittest.main()
