"""Tests for the bounded synthetic-only external VLM matrix."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from tools.expert_vlm_matrix import _load_profiles, run_matrix


def profiles():
    return {"profiles": [
        {"name": "model-b", "provider": "gemini_generate_content",
         "url": "https://b.example/v1", "model": "vision-b",
         "key_env": "LABELAGENT_B_KEY"},
        {"name": "model-a", "provider": "openai_chat",
         "url": "https://a.example/v1", "model": "vision-a",
         "key_env": "LABELAGENT_A_KEY"},
    ]}


class ExpertVlmMatrixTests(unittest.TestCase):
    def test_dry_run_is_deterministic_and_preserves_safety_boundaries(self):
        first = run_matrix(profiles(), environ={})
        second = run_matrix(copy.deepcopy(profiles()), environ={})
        self.assertEqual(first, second)
        self.assertEqual(first["mode"], "dry_run")
        self.assertEqual(first["profile_count"], 2)
        self.assertEqual(first["connections_tested"], 0)
        self.assertEqual(first["external_uploads_performed"], 0)
        self.assertFalse(first["factory_data_uploaded"])
        self.assertFalse(first["test24_used"])
        self.assertTrue(first["candidate_only"])
        self.assertFalse(first["human_truth"])
        self.assertFalse(first["approved_for_training"])
        self.assertEqual(first["deployment_decision"], "not_evaluated")
        self.assertFalse(first["quality_improvement_claimed"])
        self.assertEqual([item["name"] for item in first["results"]],
                         ["model-a", "model-b"])

    def test_execute_requires_acknowledgement_before_any_call(self):
        calls = []
        with self.assertRaisesRegex(ValueError, "acknowledgement"):
            run_matrix(profiles(), execute=True,
                       smoke_fn=lambda **kwargs: calls.append(kwargs))
        self.assertEqual(calls, [])

    def test_execute_collects_success_and_failure_without_raw_outputs(self):
        calls = []
        def smoke(**kwargs):
            calls.append(kwargs)
            if kwargs["model"] == "vision-b":
                raise TimeoutError("bounded timeout")
            return {
                "connection_tested": True,
                "external_upload_performed": True,
                "factory_data_uploaded": False,
                "status": "passed",
                "latency_ms": 125.0,
            }
        result = run_matrix(
            profiles(), execute=True, acknowledge_synthetic_upload=True,
            environ={"LABELAGENT_A_KEY": "secret-value-a",
                     "LABELAGENT_B_KEY": "secret-value-b"},
            smoke_fn=smoke)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["successful_profiles"], 1)
        self.assertEqual(result["failed_profiles"], 1)
        self.assertEqual(result["connections_tested"], 1)
        self.assertEqual(result["external_uploads_performed"], 1)
        self.assertEqual(result["uploaded_content"],
                         "generated_synthetic_images_only")
        self.assertNotIn("LABELAGENT_A_KEY", json.dumps(result))
        self.assertNotIn("secret-value-a", json.dumps(result))

    def test_secret_material_is_removed_from_receipts_and_errors(self):
        def leak(**kwargs):
            if kwargs["model"] == "vision-a":
                return {"status": "secret-value-a"}
            raise RuntimeError("upstream echoed secret-value-b")
        result = run_matrix(
            profiles(), execute=True, acknowledge_synthetic_upload=True,
            environ={"LABELAGENT_A_KEY": "secret-value-a",
                     "LABELAGENT_B_KEY": "secret-value-b"},
            smoke_fn=leak)
        serialized = json.dumps(result)
        self.assertNotIn("secret-value-a", serialized)
        self.assertNotIn("secret-value-b", serialized)
        self.assertEqual(result["failed_profiles"], 2)
        self.assertIn("credential material", result["results"][0]["error"])
        self.assertIn("[REDACTED]", result["results"][1]["error"])

    def test_rejects_raw_credentials_test24_duplicates_and_extra_fields(self):
        raw = profiles()
        raw["profiles"][0]["api_key"] = "secret"
        with self.assertRaisesRegex(ValueError, "raw credentials"):
            _load_profiles(raw)
        heldout = profiles()
        heldout["profiles"][0]["name"] = "test24-expert"
        with self.assertRaisesRegex(ValueError, "test24"):
            _load_profiles(heldout)
        duplicate = profiles()
        duplicate["profiles"][1]["name"] = "model-b"
        with self.assertRaisesRegex(ValueError, "duplicated"):
            _load_profiles(duplicate)
        extra = profiles()
        extra["profiles"][0]["image_path"] = "factory.png"
        with self.assertRaisesRegex(ValueError, "fields are invalid"):
            _load_profiles(extra)

    def test_profile_file_is_bounded_and_read_only(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "profiles.json"
            path.write_text(json.dumps(profiles()), encoding="utf-8")
            before = path.read_bytes()
            loaded = _load_profiles(path)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
