import importlib.util
from pathlib import Path
import unittest


PATH = (Path(__file__).resolve().parents[1] / "tools"
        / "benchmark_structured_decoder_overhead.py")
SPEC = importlib.util.spec_from_file_location(
    "benchmark_structured_decoder_overhead", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class StructuredDecoderOverheadTests(unittest.TestCase):
    def test_response_record_accepts_exact_target_value(self):
        target = {"assessments": {"floor": {"evidence": "present"}}}
        response = {
            "choices": [{"message": {"content": (
                '{"assessments":{"floor":{"evidence":"present"}}}')},
                "finish_reason": "stop"}],
            "local_vlm": {"cache_hit": False, "inference_stats": {
                "structured_output": True,
                "generation_seconds": 1.25,
                "constraint_setup_seconds": 0.01,
                "completion_tokens": 12,
                "prompt_tokens": 34,
            }},
        }

        record = MODULE.response_record(response, target, 1300.0)

        self.assertTrue(record["exact_target_value"])
        self.assertFalse(record["cache_hit"])
        self.assertTrue(record["structured_output"])
        self.assertEqual(record["completion_tokens"], 12)

    def test_response_record_rejects_non_json_or_changed_value(self):
        target = {"label": "floor"}
        invalid = {"choices": [{"message": {"content": "not-json"}}]}
        changed = {"choices": [{"message": {
            "content": '{"label":"slope"}'}}]}

        self.assertFalse(MODULE.response_record(
            invalid, target, 10.0)["exact_target_value"])
        self.assertFalse(MODULE.response_record(
            changed, target, 10.0)["exact_target_value"])


if __name__ == "__main__":
    unittest.main()
