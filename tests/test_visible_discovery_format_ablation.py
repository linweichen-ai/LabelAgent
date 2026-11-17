import importlib.util
from pathlib import Path
import unittest


PATH = (Path(__file__).resolve().parents[1] / "tools"
        / "evaluate_visible_discovery_format_ablation.py")
SPEC = importlib.util.spec_from_file_location(
    "evaluate_visible_discovery_format_ablation", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class VisibleDiscoveryFormatAblationTests(unittest.TestCase):
    def test_known_states_accepts_only_the_production_assessment_envelope(self):
        labels = ["floor", "slope"]
        value = {"assessments": {
            "floor": {"evidence": "present"},
            "slope": {"evidence": "absent"},
        }, "unknown_candidates": []}
        self.assertEqual(MODULE.known_states(value, labels), {
            "floor": "present", "slope": "absent"})

    def test_flat_enum_object_is_json_but_not_the_production_protocol(self):
        labels = ["floor", "slope"]
        self.assertIsNone(MODULE.known_states({
            "floor": "present", "slope": "absent"}, labels))


if __name__ == "__main__":
    unittest.main()
