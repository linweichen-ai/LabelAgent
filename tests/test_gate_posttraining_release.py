import copy
import unittest

from tools.gate_posttraining_release import gate_posttraining_release


def valid_manifest():
    dataset_hash = "a" * 64
    protocol_hash = "b" * 64
    sample_hash = "c" * 64
    adapter_hash = "d" * 64
    evaluation = {
        "status": "complete",
        "evaluation_only": True,
        "protocol_sha256": protocol_hash,
        "sample_manifest_sha256": sample_hash,
        "human_reference": True,
        "test24_returned_to_training": False,
    }
    return {
        "boundaries": {
            "test24_evaluation_only": True,
            "test24_used_for_training": False,
            "machine_candidates_are_human_truth": False,
            "production_deployed": False,
        },
        "dataset": {
            "sample_count": 34,
            "human_reviewed_count": 34,
            "approved_for_training_count": 34,
            "candidate_only": False,
            "human_truth": True,
            "contains_test24": False,
            "manifest_sha256": dataset_hash,
        },
        "training": {
            "status": "training_complete",
            "dataset_manifest_sha256": dataset_hash,
            "config_sha256": "e" * 64,
            "log_sha256": "f" * 64,
            "adapter_sha256": adapter_hash,
            "steps": 120,
            "gpu": {
                "device_count": 2,
                "devices": ["RTX 5090", "RTX 5090"],
                "max_memory_mib": 23800,
            },
        },
        "baseline_eval": {
            **evaluation,
            "metrics": {"overall_f1": .71, "slope_false_positives": 3},
        },
        "candidate_eval": {
            **evaluation,
            "adapter_sha256": adapter_hash,
            "metrics": {"overall_f1": .75, "slope_false_positives": 2},
        },
        "thresholds": {
            "min_overall_f1_delta": .01,
            "max_slope_fp_delta": 0,
        },
        "rollback": {
            "incumbent_artifact_sha256": "1" * 64,
            "restore_command": "restore-approved-incumbent",
            "verified": True,
        },
    }


class PosttrainingReleaseGateTest(unittest.TestCase):
    def test_valid_evidence_only_enables_isolated_review(self):
        report = gate_posttraining_release(valid_manifest())
        self.assertTrue(report["ready"])
        self.assertEqual(report["status"],
                         "eligible_for_isolated_release_review")
        self.assertFalse(report["production_deployment_allowed"])
        self.assertAlmostEqual(report["summary"]["overall_f1_delta"], .04)

    def test_candidate_only_training_data_fails_closed(self):
        manifest = valid_manifest()
        manifest["dataset"].update({
            "candidate_only": True,
            "human_truth": False,
            "human_reviewed_count": 0,
            "approved_for_training_count": 0,
        })
        report = gate_posttraining_release(manifest)
        self.assertFalse(report["ready"])
        self.assertIn("candidate-only data cannot pass the training gate",
                      report["errors"])

    def test_test24_leakage_fails_closed(self):
        manifest = valid_manifest()
        manifest["boundaries"]["test24_used_for_training"] = True
        manifest["dataset"]["contains_test24"] = True
        report = gate_posttraining_release(manifest)
        self.assertFalse(report["ready"])
        self.assertIn("test24 leaked into training", report["errors"])

    def test_protocol_mismatch_fails_closed(self):
        manifest = valid_manifest()
        manifest["candidate_eval"]["protocol_sha256"] = "9" * 64
        report = gate_posttraining_release(manifest)
        self.assertIn("baseline and candidate use different protocols",
                      report["errors"])

    def test_quality_regression_fails_closed(self):
        manifest = valid_manifest()
        manifest["candidate_eval"]["metrics"] = {
            "overall_f1": .70,
            "slope_false_positives": 5,
        }
        report = gate_posttraining_release(manifest)
        self.assertIn("overall F1 improvement does not meet the threshold",
                      report["errors"])
        self.assertIn("slope false positives exceed the guard",
                      report["errors"])

    def test_missing_gpu_or_rollback_evidence_fails_closed(self):
        manifest = copy.deepcopy(valid_manifest())
        manifest["training"]["gpu"] = {}
        manifest["rollback"]["verified"] = False
        report = gate_posttraining_release(manifest)
        self.assertIn("GPU usage receipt is missing", report["errors"])
        self.assertIn("rollback path has not been verified", report["errors"])


if __name__ == "__main__":
    unittest.main()
