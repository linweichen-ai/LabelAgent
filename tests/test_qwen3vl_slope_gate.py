import unittest

from tools.gate_qwen3vl_slope_candidate import gate


def score(predictions):
    visible = [False, False, True, True, False, True, True, False]
    tp = fp = fn = tn = uncertain = 0
    details = []
    for index, (expected, prediction) in enumerate(zip(visible, predictions)):
        if prediction == "uncertain":
            uncertain += 1
        elif prediction == "present" and expected:
            tp += 1
        elif prediction == "present":
            fp += 1
        elif expected:
            fn += 1
        else:
            tn += 1
        details.append({
            "sample_id": f"s{index}",
            "assistant_visible_slope_candidate": expected,
            "candidate_evidence": prediction,
        })
    return {
        "schema_version": "labelagent.qwen3vl-slope-review-score.v1",
        "human_truth": False, "human_confirmed": False,
        "production_deployed": False,
        "summary": {
            "true_positive_candidates": tp,
            "false_positive_candidates": fp,
            "false_negative_candidates": fn,
            "true_negative_candidates": tn,
            "uncertain_candidates": uncertain,
            "details": details,
        },
    }


class Qwen3VlSlopeCandidateGateTests(unittest.TestCase):
    def setUp(self):
        self.baseline = score([
            "absent", "absent", "absent", "absent",
            "absent", "absent", "present", "absent"])

    def test_passes_recall_gain_only_when_false_positives_stay_zero(self):
        candidate = score([
            "absent", "absent", "present", "absent",
            "absent", "present", "present", "absent"])
        result = gate(self.baseline, candidate)
        self.assertTrue(result["passed"])
        self.assertFalse(result["production_approved"])
        self.assertFalse(result["test24_authorized"])

    def test_rejects_recall_gain_that_adds_floor_false_positives(self):
        candidate = score([
            "present", "absent", "present", "present",
            "absent", "present", "present", "absent"])
        result = gate(self.baseline, candidate)
        self.assertFalse(result["passed"])
        self.assertFalse(result["guards"]["false_positives_not_increased"])

    def test_rejects_changed_assistant_review_or_promoted_truth(self):
        candidate = score([
            "absent", "absent", "present", "absent",
            "absent", "present", "present", "absent"])
        candidate["summary"]["details"][0][
            "assistant_visible_slope_candidate"] = True
        with self.assertRaises(ValueError):
            gate(self.baseline, candidate)
        candidate = score(["absent"] * 8)
        candidate["human_truth"] = True
        with self.assertRaises(ValueError):
            gate(self.baseline, candidate)


if __name__ == "__main__":
    unittest.main()
