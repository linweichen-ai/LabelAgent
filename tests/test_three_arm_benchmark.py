import copy
import unittest

from source.agent.three_arm_benchmark import ARMS, LABELS, build_three_arm_report


def sample(sample_id, present, pallets):
    return {"sample_id": sample_id,
            "class_presence": {label: "present" if label in present else "absent"
                               for label in LABELS},
            "pallet_instance_count": pallets}


def prediction(labels, pallets, holes=0, merges=0, conflicts=0):
    return {"present_labels": labels, "pallet_instance_count": pallets,
            "pallet_hole_errors": holes, "pallet_merge_errors": merges,
            "vehicle_robot_conflicts": conflicts}


def payload():
    references = [sample("positive", ["floor", "slope", "pallet"], 2),
                  sample("negative", ["floor", "person"], 0)]
    runs = []
    results = {
        "local_only": [prediction(["floor", "pallet"], 1, holes=1),
                       prediction(["floor", "slope", "person"], 0)],
        "all_external": [prediction(["floor", "slope", "pallet"], 2),
                         prediction(["floor", "person"], 0)],
        "hybrid": [prediction(["floor", "slope", "pallet"], 2),
                   prediction(["floor", "person"], 0)],
    }
    for arm in ARMS:
        for index, reference in enumerate(references):
            external = arm == "all_external" or (arm == "hybrid" and index == 0)
            runs.append({
                "arm": arm, "sample_id": reference["sample_id"], "status": "passed",
                "external_called": external, "fallback_occurred": False,
                "timeout": False, "latency_ms": 100 + index * 10 + ARMS.index(arm) * 100,
                "estimated_cost_usd": 0.01 if external else 0,
                "sam_artifact_sha256": f"{index + 1:064x}",
                "candidate_only": True, "human_truth": False,
                "approved_for_training": False, "prediction": results[arm][index],
            })
    return {"schema_version": "labelagent.three-arm-eval-input.v1",
            "partition": "dev_hard34", "test24": False,
            "synthetic_fixture": True, "human_reference": False,
            "reviewer_confirmed": False, "samples": references, "runs": runs}


class ThreeArmBenchmarkTest(unittest.TestCase):
    def test_reports_quality_operations_cost_and_sam_reuse(self):
        report = build_three_arm_report(payload())
        self.assertEqual([item["arm"] for item in report["arms"]], list(ARMS))
        local, external, hybrid = report["arms"]
        self.assertEqual(local["slope_positive_recall"], 0.0)
        self.assertEqual(local["class_metrics"]["slope"]["f1"], 0.0)
        self.assertEqual(local["slope_false_positive_rate_on_floor"], 1.0)
        self.assertEqual(local["pallet_instance_support_rate"], 0.5)
        self.assertEqual(local["pallet_hole_errors"], 1)
        self.assertEqual(external["micro_f1"], 1.0)
        self.assertEqual(external["external_call_ratio"], 1.0)
        self.assertEqual(hybrid["external_call_ratio"], 0.5)
        self.assertTrue(all(item["sam_artifacts_reused_across_arms"]
                            for item in report["arms"]))
        self.assertFalse(report["real_world_quality_evidence"])
        self.assertFalse(report["automatic_winner_selected"])

    def test_failed_run_is_penalized_and_counted(self):
        item = payload()
        failed = next(run for run in item["runs"]
                      if run["arm"] == "hybrid" and run["sample_id"] == "positive")
        failed.update(status="failed", timeout=True)
        failed.pop("prediction")
        report = build_three_arm_report(item)
        hybrid = report["arms"][2]
        self.assertEqual(hybrid["failed_samples"], 1)
        self.assertEqual(hybrid["timeout_count"], 1)
        self.assertEqual(hybrid["slope_positive_recall"], 0.0)
        self.assertEqual(hybrid["pallet_missed_instances"], 2)

    def test_rejects_test24_and_unconfirmed_real_reference(self):
        for mutate in (
                lambda item: item.update(partition="test24"),
                lambda item: item.update(test24=True),
                lambda item: item.update(synthetic_fixture=False),
                lambda item: item.update(synthetic_fixture=False, human_reference=True)):
            item = payload(); mutate(item)
            with self.assertRaises(ValueError):
                build_three_arm_report(item)

    def test_requires_exact_three_arm_coverage(self):
        missing = payload(); missing["runs"].pop()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            build_three_arm_report(missing)
        duplicate = payload(); duplicate["runs"][-1] = copy.deepcopy(duplicate["runs"][0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_three_arm_report(duplicate)

    def test_rejects_sam_recompute_across_arms(self):
        item = payload()
        target = next(run for run in item["runs"]
                      if run["arm"] == "hybrid" and run["sample_id"] == "positive")
        target["sam_artifact_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "not reused"):
            build_three_arm_report(item)

    def test_rejects_arm_contract_and_candidate_boundary_violations(self):
        local = payload(); local["runs"][0]["external_called"] = True
        with self.assertRaisesRegex(ValueError, "local_only"):
            build_three_arm_report(local)
        external = payload()
        run = next(row for row in external["runs"] if row["arm"] == "all_external")
        run["external_called"] = False
        with self.assertRaisesRegex(ValueError, "all_external"):
            build_three_arm_report(external)
        unsafe = payload(); unsafe["runs"][0]["approved_for_training"] = True
        with self.assertRaisesRegex(ValueError, "candidate-only"):
            build_three_arm_report(unsafe)


if __name__ == "__main__":
    unittest.main()
