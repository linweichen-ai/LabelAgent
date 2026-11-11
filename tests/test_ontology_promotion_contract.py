import copy
import json
from pathlib import Path
import tempfile
import unittest

from jsonschema import Draft202012Validator, FormatChecker

from source.agent.factory_data_contract import digest
from source.agent.ontology_promotion_contract import (
    AppendOnlyOntologyLedger,
    build_cluster_proposal,
    build_label_observation,
    build_ontology_ledger,
    evaluate_ontology_promotion,
    validate_cluster_proposal,
    validate_label_observation,
    validate_ontology_ledger,
    validate_promotion_event,
)
from tools.build_ontology_promotion_static_acceptance import (
    approval_request,
    build_static_acceptance,
    observation_payload,
)


ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class OntologyPromotionContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = load(ROOT / "configs/labelagent_label_registry.json")

    def observation(self, **kwargs):
        return build_label_observation(
            observation_payload(**kwargs), self.registry,
            observed_at="2026-11-11T20:00:00+08:00")

    def cluster(self, observations=None):
        observations = observations or [self.observation()]
        return build_cluster_proposal(
            observations, cluster_id="cluster-test-001",
            config={"algorithm": "deterministic-test", "version": "v1",
                    "config_sha256": digest(b"cluster-config")},
            proposed_at="2026-11-11T20:01:00+08:00")

    def test_module_has_no_model_gpu_network_or_remote_client(self):
        source = (ROOT / "source/agent/ontology_promotion_contract.py").read_text(encoding="utf-8")
        for forbidden in ("import torch", "import requests", "import socket",
                          "import subprocess", "import oss2", "cvat_sdk"):
            self.assertNotIn(forbidden, source)

    def test_multiview_unknown_binds_all_inputs_and_requires_human_review(self):
        event = self.observation()
        validate_label_observation(event)
        self.assertGreaterEqual(len(event["inputs"]["views"]), 2)
        for key in ("candidate", "model", "prompt"):
            self.assertIn("sha256", event["inputs"][key])
        self.assertEqual(event["result"]["outcome"], "unknown_candidate")
        self.assertTrue(event["routing"]["human_review_required"])
        self.assertFalse(event["ontology_mutation_allowed"])
        self.assertFalse(event["truth_promotion_allowed"])

    def test_known_high_confidence_consistent_can_remain_candidate_only(self):
        payload = observation_payload(outcome="known_label", confidence=0.95,
                                      sam_relation="consistent")
        payload["result"]["proposed_name"] = "pallet"
        event = build_label_observation(payload, self.registry,
                                        observed_at="2026-11-11T20:00:00+08:00")
        self.assertFalse(event["routing"]["human_review_required"])
        self.assertEqual(event["routing"]["next_state"], "known_label_candidate")
        self.assertFalse(event["truth_promotion_allowed"])

    def test_low_confidence_or_sam_conflict_always_routes_to_human(self):
        cases = (
            dict(outcome="known_label", confidence=0.4, sam_relation="consistent"),
            dict(outcome="known_label", confidence=0.95, sam_relation="conflict"),
            dict(outcome="uncertain_conflict", confidence=0.95, sam_relation="insufficient_evidence"),
        )
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                payload = observation_payload(**kwargs)
                payload["result"]["proposed_name"] = "pallet"
                event = build_label_observation(payload, self.registry,
                                                observed_at="2026-11-11T20:00:00+08:00")
                self.assertTrue(event["routing"]["human_review_required"])
                self.assertEqual(event["routing"]["next_state"], "human_review_required")

    def test_multiview_evidence_cannot_reference_unknown_artifact(self):
        event = self.observation()
        event["result"]["evidence"]["view_artifact_ids"].append(digest(b"unknown"))
        event["event_id"] = digest({key: value for key, value in event.items() if key != "event_id"})
        with self.assertRaisesRegex(ValueError, "unknown views"):
            validate_label_observation(event)

    def test_cluster_is_deduplicated_proposal_with_deterministic_representative(self):
        real = self.observation()
        sim = self.observation(source_kind="simulation", confidence=0.61,
                               sam_relation="insufficient_evidence")
        cluster = self.cluster([real, sim])
        validate_cluster_proposal(cluster)
        self.assertEqual(cluster["status"], "proposal_only")
        self.assertTrue(cluster["human_review_required"])
        self.assertFalse(cluster["ontology_mutation_allowed"])
        flattened = sum(cluster["source_partition_event_ids"].values(), [])
        self.assertCountEqual(flattened, cluster["member_event_ids"])
        self.assertEqual(cluster["representative"]["event_id"], real["event_id"])

    def test_cluster_rejects_auto_acceptable_known_label(self):
        payload = observation_payload(outcome="known_label", confidence=0.95,
                                      sam_relation="consistent")
        payload["result"]["proposed_name"] = "person"
        known = build_label_observation(payload, self.registry,
                                        observed_at="2026-11-11T20:00:00+08:00")
        with self.assertRaisesRegex(ValueError, "cannot enter unknown clustering"):
            self.cluster([known])

    def test_valid_human_approval_creates_new_registry_and_unexecuted_replay(self):
        cluster = self.cluster()
        request = approval_request(self.registry, cluster)
        event = evaluate_ontology_promotion(
            self.registry, cluster, request,
            occurred_at="2026-11-11T20:02:00+08:00")
        validate_promotion_event(event)
        self.assertEqual(event["event_type"], "ontology.registry.version.promoted")
        self.assertEqual(len(self.registry["labels"]), 7)
        self.assertEqual(len(event["new_registry"]["labels"]), 8)
        self.assertNotEqual(event["new_registry"]["registry_version"], self.registry["registry_version"])
        self.assertEqual(event["replay_plan"]["execution_status"], "planned_not_executed")
        self.assertFalse(event["replay_plan"]["automatic_truth_promotion"])

    def test_floor_slope_and_traversable_union_survive_registry_promotion(self):
        cluster = self.cluster()
        event = evaluate_ontology_promotion(
            self.registry, cluster, approval_request(self.registry, cluster),
            occurred_at="2026-11-11T20:02:00+08:00")
        names = [item["name"] for item in event["new_registry"]["labels"]]
        self.assertIn("floor", names)
        self.assertIn("slope", names)
        self.assertTrue(event["new_registry"]["training_semantics"]["floor_slope_distinct"])
        self.assertEqual(event["new_registry"]["downstream_mapping"]["source_labels"], ["floor", "slope"])

    def test_fail_closed_promotion_cases(self):
        cluster = self.cluster()
        mutations = {
            "machine_approval": lambda value: value["human_decision"].update(identity_assurance="machine"),
            "missing_definition": lambda value: value["human_decision"].update(definition=""),
            "duplicate_id": lambda value: value["human_decision"].update(class_id=1),
            "duplicate_name": lambda value: value["human_decision"].update(canonical_name="pallet"),
            "regression_failed": lambda value: value["regression_validation"].update(all_required_gates_passed=False),
            "real_metrics_missing": lambda value: value["regression_validation"].update(real_metrics={}),
            "truth_versions_mixed": lambda value: value["regression_validation"].update(truth_versions={"real": "same", "simulation": "same"}),
            "parent_drift": lambda value: value["version"].update(parent_registry_sha256=digest(b"other")),
            "same_version": lambda value: value["version"].update(registry_version=self.registry["registry_version"]),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                request = approval_request(self.registry, cluster)
                mutate(request)
                event = evaluate_ontology_promotion(
                    self.registry, cluster, request,
                    occurred_at="2026-11-11T20:02:00+08:00")
                self.assertEqual(event["event_type"], "ontology.promotion.rejected")
                self.assertFalse(event["registry_version_created"])
                self.assertFalse(event["old_registry_modified"])

    def test_append_only_ledger_idempotency_tamper_and_unique_registry(self):
        observation = self.observation()
        cluster = self.cluster([observation])
        promotion = evaluate_ontology_promotion(
            self.registry, cluster, approval_request(self.registry, cluster),
            occurred_at="2026-11-11T20:02:00+08:00")
        snapshot = build_ontology_ledger(
            [observation, cluster, promotion], ledger_id="test-ledger",
            generated_at="2026-11-11T20:03:00+08:00",
            sample_kind="synthetic_contract_fixture")
        validate_ontology_ledger(snapshot)
        tampered = copy.deepcopy(snapshot)
        tampered["records"][0]["event"]["ontology_mutation_allowed"] = True
        with self.assertRaises(ValueError):
            validate_ontology_ledger(tampered)
        with tempfile.TemporaryDirectory() as directory:
            ledger = AppendOnlyOntologyLedger(directory, "runtime-ledger")
            first, replay = ledger.append(observation)
            self.assertFalse(replay)
            second, replay = ledger.append(observation)
            self.assertTrue(replay)
            self.assertEqual(first, second)
            ledger.append(cluster)
            ledger.append(promotion)
            duplicate = copy.deepcopy(promotion)
            duplicate["idempotency_key"] = digest(b"new-idempotency")
            duplicate["event_id"] = digest({key: value for key, value in duplicate.items() if key != "event_id"})
            with self.assertRaisesRegex(ValueError, "already exists"):
                ledger.append(duplicate)

    def test_promotion_event_rejects_posthoc_human_or_metric_tampering(self):
        cluster = self.cluster()
        event = evaluate_ontology_promotion(
            self.registry, cluster, approval_request(self.registry, cluster),
            occurred_at="2026-11-11T20:02:00+08:00")
        for mutate in (
            lambda value: value["human_decision"].update(reviewer_id=""),
            lambda value: value["regression_validation"]["real_metrics"].update(precision=1.5),
            lambda value: value["regression_validation"].update(
                truth_versions={"real": "same", "simulation": "same"}),
        ):
            tampered = copy.deepcopy(event)
            mutate(tampered)
            tampered["event_id"] = digest({key: value for key, value in tampered.items()
                                           if key != "event_id"})
            with self.assertRaises(ValueError):
                validate_promotion_event(tampered)

    def test_static_examples_and_exact_emitted_files_validate(self):
        cases = (
            ("examples/vlm_unknown_candidate_multiview_static.json", "schemas/vlm-label-observation.schema.json"),
            ("examples/ontology_unknown_cluster_proposal_static.json", "schemas/ontology-cluster-proposal.schema.json"),
            ("examples/ontology_registry_promotion_positive_static.json", "schemas/ontology-registry-promotion.schema.json"),
            ("examples/ontology_promotion_machine_rejected_static.json", "schemas/ontology-promotion-rejection.schema.json"),
            ("docs/test-results/ontology-promotion-audit-ledger-static-20261111.json", "schemas/ontology-audit-ledger.schema.json"),
            ("docs/test-results/ontology-promotion-static-validation-20261111.json", "schemas/ontology-static-validation-receipt.schema.json"),
        )
        for document, schema in cases:
            with self.subTest(document=document):
                validator = Draft202012Validator(load(ROOT / schema), format_checker=FormatChecker())
                errors = list(validator.iter_errors(load(ROOT / document)))
                self.assertEqual([], errors)

    def test_static_builder_is_deterministic_and_nonproduction(self):
        first, second = build_static_acceptance(), build_static_acceptance()
        self.assertEqual(first, second)
        receipt = load(ROOT / "docs/test-results/ontology-promotion-static-validation-20261111.json")
        self.assertEqual(receipt["production_registry_versions_created"], 0)
        self.assertEqual(receipt["production_replays_executed"], 0)
        self.assertEqual(receipt["cnb_status"], "pending_not_contacted")
        self.assertFalse(receipt["execution_boundary"]["vlm_called"])
        self.assertFalse(receipt["execution_boundary"]["gpu_called"])
        for relative, expected in receipt["artifact_content_sha256"].items():
            self.assertEqual(digest(load(ROOT / relative)), expected, relative)


if __name__ == "__main__":
    unittest.main()
