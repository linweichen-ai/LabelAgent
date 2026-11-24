import json
import tempfile
import unittest
from pathlib import Path

from tests.test_build_mixed_hard_adjudication_workspace import (
    MixedHardAdjudicationWorkspaceTest,
)
from tools.audit_second_assistant_slope_triage import (
    _sha256,
    audit_second_assistant_triage,
)
from tools.build_mixed_hard_adjudication_workspace import validate_packet


class SecondAssistantSlopeTriageAuditTest(unittest.TestCase):
    def fixture(self, root: Path):
        packet = MixedHardAdjudicationWorkspaceTest().packet(root / "packet")
        manifest, digest = validate_packet(packet)
        sample = manifest["rows"][0]
        qwen = root / "qwen.json"
        qwen.write_text(json.dumps({
            "rows": [{
                "sample_id": sample["sample_id"],
                "suggestion": {"assessments": {"slope": "uncertain"}},
            }],
        }), encoding="utf-8")
        triage = root / "triage.json"
        triage.write_text(json.dumps({
            "schema_version": "labelagent.local-second-assistant-slope-triage.v1",
            "status": "assistant_review_complete_human_confirmation_pending",
            "review_manifest_sha256": digest,
            "local_qwen_suggestions_sha256": _sha256(qwen),
            "reviewed_sample_count": len(manifest["rows"]),
            "priority_disagreement_count": 1,
            "rows": [{
                "sample_id": sample["sample_id"],
                "source_image_sha256": sample["source_image_sha256"],
                "local_qwen_slope": "uncertain",
                "assistant_slope_candidate": "present",
                "assistant_confidence": "low",
                "difficulty_flags": ["edge"],
                "rationale": "partial transition",
            }],
            "candidate_only": True,
            "human_truth": False,
            "approved_for_training": False,
            "training_authorized": False,
            "automatic_approval": False,
            "test24_used": False,
            "external_api_calls": 0,
            "factory_data_uploaded": False,
            "production_deployed": False,
            "human_review_required": True,
        }), encoding="utf-8")
        return packet, qwen, triage

    def test_accepts_hash_bound_candidate_only_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            packet, qwen, triage = self.fixture(Path(directory))
            audit = audit_second_assistant_triage(packet, qwen, triage)
            self.assertEqual(audit["priority_disagreement_count"], 1)
            self.assertEqual(audit["qwen_slope_states"], {"uncertain": 1})
            self.assertFalse(audit["human_truth"])
            self.assertTrue(audit["human_review_required"])

    def test_rejects_hash_drift_and_training_promotion(self):
        with tempfile.TemporaryDirectory() as directory:
            packet, qwen, triage = self.fixture(Path(directory))
            payload = json.loads(triage.read_text(encoding="utf-8"))
            payload["rows"][0]["source_image_sha256"] = "0" * 64
            triage.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source image binding"):
                audit_second_assistant_triage(packet, qwen, triage)
            packet, qwen, triage = self.fixture(Path(directory) / "second")
            payload = json.loads(triage.read_text(encoding="utf-8"))
            payload["approved_for_training"] = True
            triage.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "approved_for_training"):
                audit_second_assistant_triage(packet, qwen, triage)


if __name__ == "__main__":
    unittest.main()
