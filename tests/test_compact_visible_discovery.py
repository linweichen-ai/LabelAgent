import importlib.util
from pathlib import Path
import unittest


PATH = (Path(__file__).resolve().parents[1] / "tools"
        / "evaluate_compact_visible_discovery.py")
SPEC = importlib.util.spec_from_file_location(
    "evaluate_compact_visible_discovery", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CompactVisibleDiscoveryTests(unittest.TestCase):
    def test_schema_requires_every_registry_label_and_open_set_tail(self):
        schema = MODULE.compact_schema(["floor", "slope"])
        self.assertEqual(schema["required"], [
            "assessments", "unknown_candidates"])
        assessments = schema["properties"]["assessments"]
        self.assertEqual(assessments["required"], ["floor", "slope"])
        self.assertFalse(assessments["additionalProperties"])

    def test_instruction_replacement_fails_closed_on_contract_drift(self):
        with self.assertRaises(ValueError):
            MODULE.compact_instruction("changed production prompt")


if __name__ == "__main__":
    unittest.main()
