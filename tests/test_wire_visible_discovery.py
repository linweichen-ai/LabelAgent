import importlib.util
from pathlib import Path
import unittest


PATH = (Path(__file__).resolve().parents[1] / "tools"
        / "evaluate_wire_visible_discovery.py")
SPEC = importlib.util.spec_from_file_location(
    "evaluate_wire_visible_discovery", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class WireVisibleDiscoveryTests(unittest.TestCase):
    def test_wire_schema_maps_every_label_without_loss(self):
        schema, mapping = MODULE.wire_schema(["floor", "other_robot"])
        self.assertEqual(mapping, {"l0": "floor", "l1": "other_robot"})
        self.assertEqual(schema["properties"]["assessments"]["required"],
                         ["l0", "l1"])

    def test_expand_wire_restores_production_field_names(self):
        mapping = {"l0": "floor"}
        value = {"assessments": {"l0": {
            "e": "present", "c": 0.9, "r": "visible concrete surface"}},
            "unknown_candidates": []}
        self.assertEqual(MODULE.expand_wire(value, mapping), {
            "assessments": {"floor": {"evidence": "present",
                "confidence": 0.9, "reason": "visible concrete surface"}},
            "unknown_candidates": []})


if __name__ == "__main__":
    unittest.main()
