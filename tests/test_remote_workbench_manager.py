import importlib.util
from pathlib import Path
import unittest


PATH = Path(__file__).resolve().parents[1] / "tools" / "manage_remote_workbench.py"
SPEC = importlib.util.spec_from_file_location("manage_remote_workbench", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RemoteWorkbenchManagerTests(unittest.TestCase):
    def test_optional_fast_route_uses_a_short_default_wait(self):
        self.assertEqual(MODULE.optional_fast_wait_seconds({}), 3.0)

    def test_optional_fast_route_wait_is_bounded_and_invalid_values_are_safe(self):
        self.assertEqual(MODULE.optional_fast_wait_seconds({
            "LABELAGENT_FAST_ROUTE_WAIT_SECONDS": "0"}), 1.0)
        self.assertEqual(MODULE.optional_fast_wait_seconds({
            "LABELAGENT_FAST_ROUTE_WAIT_SECONDS": "60"}), 20.0)
        self.assertEqual(MODULE.optional_fast_wait_seconds({
            "LABELAGENT_FAST_ROUTE_WAIT_SECONDS": "7.5"}), 7.5)
        self.assertEqual(MODULE.optional_fast_wait_seconds({
            "LABELAGENT_FAST_ROUTE_WAIT_SECONDS": "invalid"}), 3.0)

    def test_startup_uses_the_bounded_wait_for_the_optional_route(self):
        source = PATH.read_text(encoding="utf-8")
        self.assertIn(
            'optional_fast_wait_seconds(), "5090 可选加速通道")', source)
        self.assertNotIn('20, "5090 可选加速通道")', source)


if __name__ == "__main__":
    unittest.main()
