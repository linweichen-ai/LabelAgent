import json
from pathlib import Path
import tempfile
import unittest

from tools.audit_sam3_projection_training import audit


class ProjectionTrainingAuditTests(unittest.TestCase):
    def fixture(self, root: Path):
        rounds = []
        for steps in (54, 81):
            rounds.append({
                "schema_version": "labelagent.sam3-projection-night-round.v1",
                "candidate_only": True, "pseudo_truth": True,
                "not_human_truth": True, "not_for_deployment": True,
                "deployment_performed": False, "test24_in_training": False,
                "cumulative_optimizer_steps": steps, "exit_code": 0,
                "runtime_failure": None, "checkpoint_sha256": str(steps) * 32,
                "elapsed_seconds": 30.0,
            })
        comparison = {
            "evaluation_only": True, "machine_reference_only": True,
            "test24_training_return_allowed": False, "not_human_truth": True,
            "not_for_deployment": True,
            "deployment_decision": "forbidden_without_human_truth",
        }
        values = {
            "round54": rounds[0], "round81": rounds[1],
            "compare54": comparison, "compare81": comparison,
            "threeway": {
                "evaluation_only": True, "machine_reference_only": True,
                "test24_training_steps": 0, "not_human_truth": True,
                "not_for_deployment": True, "images": 24,
                "means": {"base": {"union_iou": .97, "boundary_f1": .98},
                          "best": {"union_iou": .965, "boundary_f1": .977},
                          "current": {"union_iou": .963, "boundary_f1": .976}},
            },
            "review81": {"pseudo_truth": True, "long_training_gate": "blocked",
                         "not_human_truth": True, "not_for_deployment": True,
                         "catastrophic_frames": ["bad.png"]},
        }
        paths = {}
        for name, value in values.items():
            path = root / f"{name}.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            paths[name] = path
        return paths

    def test_negative_projection_result_keeps_incumbent(self):
        with tempfile.TemporaryDirectory() as directory:
            value = audit(self.fixture(Path(directory)))
        self.assertEqual(value["status"], "passed_negative_result_audit")
        self.assertTrue(value["decision"]["keep_incumbent_base_sam3"])
        self.assertFalse(value["decision"]["continue_pseudo_truth_long_training"])

    def test_test24_training_leak_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.fixture(Path(directory))
            value = json.loads(paths["round81"].read_text())
            value["test24_in_training"] = True
            paths["round81"].write_text(json.dumps(value))
            result = audit(paths)
        self.assertEqual(result["status"], "failed")
        self.assertIn("round2: test24 entered training", result["errors"])

    def test_missing_regression_evidence_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.fixture(Path(directory))
            value = json.loads(paths["threeway"].read_text())
            value["means"]["current"]["union_iou"] = .98
            paths["threeway"].write_text(json.dumps(value))
            result = audit(paths)
        self.assertEqual(result["status"], "failed")
        self.assertIn("expected current regression is absent", result["errors"])


if __name__ == "__main__":
    unittest.main()
