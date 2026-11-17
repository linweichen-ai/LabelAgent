import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


PATH = (Path(__file__).resolve().parents[1] / "tools"
        / "benchmark_sdpa_backends.py")
SPEC = importlib.util.spec_from_file_location("benchmark_sdpa_backends", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Output:
    def float(self):
        return self

    def mean(self):
        return self

    def item(self):
        return 0.25


class SDPABenchmarkTests(unittest.TestCase):
    def test_default_backend_records_supported_timing(self):
        calls = []
        torch_module = SimpleNamespace(cuda=SimpleNamespace(
            synchronize=lambda: calls.append("sync")))
        functional = SimpleNamespace(scaled_dot_product_attention=
            lambda *args, **kwargs: (calls.append("call") or Output()))

        result = MODULE.timed_attention(
            torch_module, functional, None, object(), object(), object(),
            None, False, warmups=1, iterations=2)

        self.assertTrue(result["supported"])
        self.assertEqual(result["output_mean"], 0.25)
        self.assertEqual(calls.count("call"), 3)

    def test_runtime_failure_is_preserved_as_evidence(self):
        torch_module = SimpleNamespace(cuda=SimpleNamespace(
            synchronize=lambda: None))

        def fail(*args, **kwargs):
            raise RuntimeError("unsupported kernel")

        result = MODULE.timed_attention(
            torch_module,
            SimpleNamespace(scaled_dot_product_attention=fail),
            None, object(), object(), object(), None, False,
            warmups=1, iterations=1)

        self.assertFalse(result["supported"])
        self.assertEqual(result["error_type"], "RuntimeError")
        self.assertIn("unsupported kernel", result["error"])


if __name__ == "__main__":
    unittest.main()
