import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest import mock

from jsonschema import Draft202012Validator, FormatChecker

from source.agent.stage0_training_contract import (
    EXPECTED_NAMES,
    EXPECTED_NAMES_SHA256,
    EXPECTED_PARAMETER_COUNT,
    compile_dry_run,
    digest,
    evaluate_deployment_candidate,
    validate_package,
)
from tools.build_sam3_stage0_static_acceptance import (
    COMPILED_AT,
    build_static_acceptance,
)
from tools.sam3_stage0_launcher import compile_only, explicit_launch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PATH = ROOT / "configs/sam3_sfl_cpd15y_stage0_training_package.json"
MODEL_CARD_PATH = ROOT / "configs/sam3_stage0_model_card.blocked.json"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def metric(iou=0.7, boundary=0.65, precision=0.8, recall=0.75, fp=5):
    return {
        "mask_iou": iou, "boundary_f1": boundary,
        "object_precision": precision, "object_recall": recall,
        "false_positives": fp,
    }


def evaluation_payload():
    side = {
        "overall": metric(),
        "per_class": {"floor": metric(iou=0.8), "slope": metric(iou=0.6)},
        "source_real": metric(iou=0.68),
        "source_simulation": metric(iou=0.72, fp=1),
    }
    return {
        "evaluation_id": "STATIC_UNIT_TEST",
        "fixed_holdout": True,
        "test24_in_training": False,
        "fixed_holdout_manifest_sha256": "1" * 64,
        "candidate_checkpoint_sha256": "2" * 64,
        "incumbent": copy.deepcopy(side),
        "candidate": copy.deepcopy(side),
    }


class Sam3Stage0TrainingPackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.package = load(PACKAGE_PATH)

    def schema_errors(self, value, name):
        schema = load(ROOT / "schemas" / name)
        return list(Draft202012Validator(
            schema, format_checker=FormatChecker()).iter_errors(value))

    def test_exact_package_is_schema_and_semantically_valid(self):
        self.assertEqual(self.schema_errors(
            self.package, "sam3-stage0-training-package.schema.json"), [])
        self.assertTrue(validate_package(self.package, ROOT))

    def test_exact_verified_scope_is_six_tensors_and_197376_parameters(self):
        scope = self.package["trainable_scope"]
        self.assertEqual(tuple(scope["exact_parameter_names"]), EXPECTED_NAMES)
        self.assertEqual(scope["tensor_count"], 6)
        self.assertEqual(scope["parameter_count"], EXPECTED_PARAMETER_COUNT)
        names_bytes = ("\n".join(EXPECTED_NAMES) + "\n").encode()
        self.assertEqual(hashlib.sha256(names_bytes).hexdigest(),
                         EXPECTED_NAMES_SHA256)

    def test_bound_recipe_scope_trainer_and_base_recipe_hashes_match(self):
        model = self.package["model"]
        for path_name, hash_name in (
            ("stage0_recipe_path", "stage0_recipe_sha256"),
            ("stage0_single_gpu_recipe_path", "stage0_single_gpu_recipe_sha256"),
            ("base_recipe_path", "base_recipe_sha256"),
            ("scope_module_path", "scope_module_sha256"),
            ("trainer_module_path", "trainer_module_sha256"),
        ):
            actual = hashlib.sha256(
                (ROOT / model[path_name]).read_bytes()).hexdigest()
            self.assertEqual(actual, model[hash_name])

    def test_checked_in_package_compiles_to_fail_closed_plan(self):
        compiled = compile_only(PACKAGE_PATH, compiled_at=COMPILED_AT)
        plan = compiled["plan"]
        self.assertEqual(plan["status"], "rejected_fail_closed")
        self.assertFalse(plan["process_created"])
        self.assertFalse(plan["gpu_called"])
        self.assertFalse(plan["automatic_launch_performed"])
        self.assertIn("genuine_human_pixel_truth_missing", plan["blockers"])
        self.assertIn("fixed_independent_holdout_baseline_missing", plan["blockers"])
        self.assertIn("explicit_training_authorization_missing", plan["blockers"])
        self.assertIn("stage0_early_stop_checkpoint_runtime_control_unverified",
                      plan["blockers"])
        self.assertIn("current_package_compile_only_process_creation_forbidden",
                      plan["blockers"])
        self.assertIsNotNone(compiled["rejection"])

    def test_second_stage_rejects_before_process_creation(self):
        compiled = compile_only(PACKAGE_PATH, compiled_at=COMPILED_AT)
        with mock.patch("tools.sam3_stage0_launcher.subprocess.Popen") as popen:
            with self.assertRaisesRegex(PermissionError, "fail-closed"):
                explicit_launch(compiled, explicit_launch=True)
            popen.assert_not_called()

    def test_launch_without_explicit_flag_is_rejected_before_popen(self):
        compiled = compile_only(PACKAGE_PATH, compiled_at=COMPILED_AT)
        with mock.patch("tools.sam3_stage0_launcher.subprocess.Popen") as popen:
            with self.assertRaisesRegex(PermissionError, "explicit-launch"):
                explicit_launch(compiled, explicit_launch=False)
            popen.assert_not_called()

    def test_machine_preannotations_can_never_be_truth(self):
        value = copy.deepcopy(self.package)
        value["data"]["machine_preannotations_truth_eligible"] = True
        with self.assertRaisesRegex(ValueError, "machine/synthetic"):
            validate_package(value, ROOT)
        self.assertNotEqual(self.schema_errors(
            value, "sam3-stage0-training-package.schema.json"), [])

    def test_synthetic_fixtures_can_never_be_truth(self):
        value = copy.deepcopy(self.package)
        value["data"]["synthetic_fixtures_truth_eligible"] = True
        with self.assertRaisesRegex(ValueError, "machine/synthetic"):
            validate_package(value, ROOT)

    def test_test24_and_fixed_holdout_can_never_enter_training(self):
        for field in ("test24_in_training", "fixed_holdout_in_training"):
            value = copy.deepcopy(self.package)
            value["data"][field] = True
            with self.assertRaises(ValueError):
                validate_package(value, ROOT)

    def test_wrong_source_hash_is_rejected(self):
        value = copy.deepcopy(self.package)
        value["model"]["stage0_recipe_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source hash mismatch"):
            validate_package(value, ROOT)

    def test_wrong_scope_name_or_count_is_rejected(self):
        for mutation in ("name", "count"):
            value = copy.deepcopy(self.package)
            if mutation == "name":
                value["trainable_scope"]["exact_parameter_names"][0] += ".drift"
            else:
                value["trainable_scope"]["parameter_count"] += 1
            with self.assertRaisesRegex(ValueError, "scope drift"):
                validate_package(value, ROOT)

    def test_single_and_dual_profiles_have_equal_effective_batch(self):
        topology = self.package["topology"]
        self.assertEqual(topology["single_gpu"]["effective_global_batch"], 2)
        self.assertEqual(topology["dual_gpu"]["effective_global_batch"], 2)
        self.assertEqual(topology["dual_gpu"]["strategy"], "synchronous_ddp")
        self.assertEqual(topology["dual_gpu"]["backend"], "nccl")
        single_recipe = (ROOT / self.package["model"][
            "stage0_single_gpu_recipe_path"]).read_text(encoding="utf-8")
        self.assertIn("gradient_accumulation_steps: 2", single_recipe)
        self.assertIn("gpus_per_node: 1", single_recipe)

    def test_early_stopping_never_tunes_on_fixed_holdout_or_test24(self):
        package_text = PACKAGE_PATH.read_text(encoding="utf-8")
        early = self.package["optimization"]["early_stopping"]
        self.assertEqual(early["metric"],
                         "train72_sequence_disjoint_val.mask_iou")
        self.assertFalse(early["fixed_holdout_used"])
        self.assertNotIn("holdout.mask_iou", package_text)
        self.assertNotIn("test24", early["metric"])

    def test_recipe_is_segmentation_enabled_and_never_mentions_test24(self):
        recipe = (ROOT / self.package["model"]["stage0_recipe_path"]
                  ).read_text(encoding="utf-8")
        self.assertIn("enable_segmentation: true", recipe)
        self.assertIn("sam3.train.loss.loss_fns.Masks", recipe)
        self.assertIn("Stage0MaskEmbedTrainer", recipe)
        self.assertIn("SAM3_STAGE0_TRAIN_ROOT", recipe)
        self.assertIn("SAM3_STAGE0_VAL_ROOT", recipe)
        self.assertNotIn("test24", recipe.lower())

    def test_optimizer_has_exact_parameter_allowlist(self):
        trainer = (ROOT / self.package["model"]["trainer_module_path"]
                   ).read_text(encoding="utf-8")
        scope = (ROOT / self.package["model"]["scope_module_path"]
                 ).read_text(encoding="utf-8")
        self.assertIn("param_allowlist=set(EXPECTED_PARAMETER_NAMES)", trainer)
        self.assertIn("requires_grad_(False)", scope)
        self.assertIn("requires_grad_(True)", scope)

    def test_blocked_model_card_is_schema_valid_and_claims_no_training(self):
        card = load(MODEL_CARD_PATH)
        self.assertEqual(self.schema_errors(
            card, "sam3-stage0-model-card.schema.json"), [])
        self.assertIsNone(card["candidate_checkpoint"])
        self.assertEqual(card["evaluation"]["status"], "not_run")
        self.assertFalse(card["evaluation"]["deployment_candidate"])
        self.assertTrue(card["rollback"]["incumbent_retained"])

    def test_deployment_candidate_requires_overall_and_slope_non_regression(self):
        payload = evaluation_payload()
        accepted = evaluate_deployment_candidate(payload)
        self.assertTrue(accepted["deployment_candidate"])
        self.assertFalse(accepted["incumbent_retained"])
        payload["candidate"]["per_class"]["slope"]["boundary_f1"] -= 0.01
        rejected = evaluate_deployment_candidate(payload)
        self.assertFalse(rejected["deployment_candidate"])
        self.assertTrue(rejected["incumbent_retained"])
        self.assertIn("slope.boundary_f1_regressed", rejected["failures"])

    def test_overall_false_positive_increase_retains_incumbent(self):
        payload = evaluation_payload()
        payload["candidate"]["overall"]["false_positives"] += 1
        decision = evaluate_deployment_candidate(payload)
        self.assertFalse(decision["deployment_candidate"])
        self.assertIn("overall.false_positives_increased", decision["failures"])

    def test_missing_real_sim_or_floor_metrics_is_rejected(self):
        for path in (("source_simulation",), ("per_class", "floor")):
            payload = evaluation_payload()
            target = payload["candidate"]
            if len(path) == 1:
                del target[path[0]]
            else:
                del target[path[0]][path[1]]
            with self.assertRaises(ValueError):
                evaluate_deployment_candidate(payload)

    def test_deterministic_static_outputs_match_exact_files_and_schemas(self):
        outputs = build_static_acceptance()
        schema_by_name = {
            "sam3-stage0-dry-run-plan-20261111.json": "sam3-stage0-dry-run-plan.schema.json",
            "sam3-stage0-launch-rejection-20261111.json": "sam3-stage0-launch-rejection.schema.json",
            "sam3_stage0_slope_regression_rejected_static.json": "sam3-stage0-deployment-decision.schema.json",
            "sam3-stage0-static-validation-20261111.json": "sam3-stage0-static-validation-receipt.schema.json",
        }
        for path, value in outputs.items():
            self.assertTrue(path.is_file(), path)
            self.assertEqual(load(path), value)
            self.assertEqual(self.schema_errors(value, schema_by_name[path.name]), [])
        receipt = next(value for path, value in outputs.items()
                       if path.name == "sam3-stage0-static-validation-20261111.json")
        expected = copy.deepcopy(receipt)
        receipt_hash = expected.pop("receipt_sha256")
        self.assertEqual(receipt_hash, digest(expected))
        self.assertFalse(receipt["training_started"])
        self.assertEqual(receipt["checkpoints_created"], 0)
        self.assertEqual(receipt["cnb_status"], "pending_not_contacted")


if __name__ == "__main__":
    unittest.main()
