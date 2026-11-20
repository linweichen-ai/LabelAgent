import copy
import unittest

from tools.audit_posttraining_readiness import audit_posttraining_readiness


def evidence():
    adapter = "a" * 64
    gate = {
        "status": "blocked_waiting_for_human_review",
        "training_authorized": False,
        "test_split_policy": "evaluation_only_never_train",
        "machine_preannotations_are_ground_truth": False,
        "checklist": {"rows": 96, "approved": 0},
        "human_exports": {"present": {"train": False, "test": False}},
        "blockers": ["human exports missing"],
    }
    receipt = {
        "status": "training_complete",
        "track": "teacher_distillation",
        "human_truth": False,
        "production_deployed": False,
        "test24_used_for_training": False,
        "samples_loaded": 62,
        "steps": 62,
        "adapter_sha256": adapter,
    }
    audit = {
        "status": "passed",
        "adapter_sha256": adapter,
        "boundaries": {
            "test24_training_false": True,
            "human_truth_false": True,
        },
    }
    external = {
        "status": "external_evaluated_not_deployed",
        "split": "test24",
        "evaluation_only": True,
        "training_return_allowed": False,
        "human_truth": False,
        "production_deployed": False,
        "adapter_sha256": adapter,
        "summary": {
            "teacher_evidence_matches": 180,
            "teacher_evidence_total": 192,
            "teacher_evidence_accuracy": .9375,
        },
    }
    hard = {
        "selected_count": 4,
        "safe_expansion_count": 0,
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "production_deployed": False,
        "reference_labels_used": False,
        "workbench_writes": 0,
    }
    return gate, receipt, audit, external, hard


class PosttrainingReadinessTest(unittest.TestCase):
    def test_teacher_lora_is_valid_without_authorizing_human_training(self):
        report = audit_posttraining_readiness(*evidence())

        self.assertFalse(report["integrity_errors"])
        self.assertEqual(
            report["status"],
            "offline_teacher_lora_exists_human_supervised_training_blocked")
        self.assertTrue(report["vlm_teacher_lora"]["training_executed"])
        self.assertFalse(report["sam3_pixel_finetune_ready"])
        self.assertEqual(
            report["vlm_teacher_lora"]["test24_teacher_evidence_accuracy"],
            .9375)

    def test_test24_leakage_fails_closed(self):
        values = list(evidence())
        values[1] = copy.deepcopy(values[1])
        values[1]["test24_used_for_training"] = True

        report = audit_posttraining_readiness(*values)

        self.assertEqual(report["status"], "invalid_or_inconsistent_evidence")
        self.assertIn("test24 leaked into LoRA training",
                      report["integrity_errors"])
        self.assertFalse(report["production_model_change_allowed"])

    def test_human_gate_can_only_enable_isolated_training(self):
        values = list(evidence())
        values[0] = copy.deepcopy(values[0])
        values[0].update({
            "status": "ready_for_isolated_training",
            "training_authorized": True,
            "checklist": {"rows": 96, "approved": 96},
            "human_exports": {"present": {"train": True, "test": True}},
            "blockers": [],
        })

        report = audit_posttraining_readiness(*values)

        self.assertEqual(
            report["status"],
            "human_supervised_training_ready_for_isolated_run")
        self.assertTrue(report["sam3_pixel_finetune_ready"])
        self.assertFalse(report["production_model_change_allowed"])


if __name__ == "__main__":
    unittest.main()
