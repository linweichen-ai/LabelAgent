import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("run_local_agent_acceptance", Path(__file__).resolve().parents[1] / "tools/run_local_agent_acceptance.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AcceptanceEvidenceTests(unittest.TestCase):
    def test_budget_exhaustion_with_successful_calls_is_not_completion(self):
        result = {"agent": {"rounds": 5, "converged": False, "stop_reason": "max_rounds"}, "shapes": []}
        traces = [{"service": service, "status": "succeeded"} for service in ("GPT", "SAM 3")]
        self.assertFalse(MODULE.evaluate_run(result, traces)["loop_completed"])

    def test_one_pass_done_is_not_evidence_of_multi_round_correction_or_accuracy(self):
        result = {"agent": {"rounds": 1, "converged": True, "stop_reason": "done"}, "shapes": []}
        traces = [{"service": service, "status": "succeeded"} for service in ("GPT", "SAM 3")]
        receipt = MODULE.evaluate_run(result, traces)
        self.assertTrue(receipt["loop_completed"])
        self.assertFalse(receipt["multi_round_correction_observed"])
        self.assertFalse(receipt["pixel_accuracy_evaluated"])
        self.assertFalse(receipt["candidates_confirmed_as_truth"])

    def test_missing_real_service_trace_prevents_completion(self):
        result = {"agent": {"rounds": 1, "converged": True, "stop_reason": "done"}, "shapes": []}
        self.assertFalse(MODULE.evaluate_run(result, [{"service": "SAM 3", "status": "succeeded"}])["loop_completed"])


if __name__ == "__main__":
    unittest.main()
