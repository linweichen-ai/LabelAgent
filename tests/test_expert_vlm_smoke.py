"""Tests for the synthetic-only external expert connectivity smoke."""
import json
import unittest

from tools.expert_vlm_smoke import run_smoke, synthetic_factory_image


class ExpertVlmSmokeTests(unittest.TestCase):
    def test_dry_run_is_deterministic_and_never_calls_or_uploads(self):
        calls = []
        kwargs = dict(
            provider="openai_responses", url="https://api.example/v1",
            model="vision-model", key_env="LABELAGENT_TEST_KEY",
            environ={}, chat_fn=lambda *args, **kwargs: calls.append(args))
        first = run_smoke(**kwargs)
        second = run_smoke(**kwargs)
        self.assertEqual(first, second)
        self.assertFalse(first["connection_tested"])
        self.assertFalse(first["external_upload_performed"])
        self.assertFalse(first["factory_data_uploaded"])
        self.assertFalse(first["ready_to_execute"])
        self.assertEqual(calls, [])
        self.assertEqual(synthetic_factory_image(), synthetic_factory_image())

    def test_synthetic_image_supports_bounded_resolution_variants(self):
        resized = synthetic_factory_image(320, 256)
        self.assertNotEqual(resized, synthetic_factory_image())
        with self.assertRaisesRegex(ValueError, "128..2048"):
            synthetic_factory_image(64, 256)

    def test_execute_requires_acknowledgement_and_credential_before_call(self):
        calls = []
        base = dict(
            provider="anthropic_messages", url="https://api.example/v1",
            model="vision-model", key_env="LABELAGENT_TEST_KEY",
            execute=True, chat_fn=lambda *args, **kwargs: calls.append(args))
        with self.assertRaisesRegex(ValueError, "acknowledgement"):
            run_smoke(**base, environ={"LABELAGENT_TEST_KEY": "secret"})
        with self.assertRaisesRegex(ValueError, "environment variable is missing"):
            run_smoke(**base, acknowledge_synthetic_upload=True, environ={})
        self.assertEqual(calls, [])

    def test_execute_records_only_synthetic_receipt_and_never_echoes_key(self):
        observed = []
        def respond(config, instruction, image, response_schema, metadata_out):
            observed.append((config, instruction, image, response_schema))
            metadata_out.update({"external_vlm": {
                "usage": {"prompt_tokens": 12, "completion_tokens": 3,
                          "total_tokens": 15}, "usage_available": True}})
            return {"status": "synthetic_image_received"}
        receipt = run_smoke(
            provider="gemini_generate_content",
            url="https://generativelanguage.googleapis.com/v1beta",
            model="gemini-test", key_env="LABELAGENT_TEST_KEY",
            execute=True, acknowledge_synthetic_upload=True,
            environ={"LABELAGENT_TEST_KEY": "secret-value"}, chat_fn=respond)
        self.assertTrue(receipt["connection_tested"])
        self.assertTrue(receipt["external_upload_performed"])
        self.assertFalse(receipt["factory_data_uploaded"])
        self.assertEqual(receipt["uploaded_content"],
                         "generated_synthetic_image_only")
        self.assertEqual(len(observed), 1)
        self.assertEqual(
            receipt["expert_metadata"]["external_vlm"]["usage"]["total_tokens"],
            15)
        self.assertEqual(observed[0][2], synthetic_factory_image())
        self.assertNotIn("secret-value", json.dumps(receipt))

    def test_cached_success_does_not_claim_a_second_external_upload(self):
        def cached(config, instruction, image, response_schema, metadata_out):
            metadata_out.update({
                "model_route": "external_expert_cache",
                "expert_cache_hit": True,
                "expert_network_dispatched": False,
            })
            return {"status": "synthetic_image_received"}

        receipt = run_smoke(
            provider="openai_chat", url="https://api.example/v1",
            model="vision-model", key_env="LABELAGENT_TEST_KEY",
            execute=True, acknowledge_synthetic_upload=True,
            environ={"LABELAGENT_TEST_KEY": "secret-value"}, chat_fn=cached)
        self.assertTrue(receipt["connection_tested"])
        self.assertFalse(receipt["external_upload_performed"])
        self.assertEqual(receipt["uploaded_content"], "none_cache_reuse")
        self.assertTrue(receipt["expert_metadata"]["expert_cache_hit"])
        self.assertNotIn("secret-value", json.dumps(receipt))

    def test_historical_ui_alias_is_canonicalized_before_dispatch(self):
        observed = []
        receipt = run_smoke(
            provider="openai", url="https://api.example/v1",
            model="vision-model", key_env="LABELAGENT_TEST_KEY",
            execute=True, acknowledge_synthetic_upload=True,
            environ={"LABELAGENT_TEST_KEY": "secret"},
            chat_fn=lambda config, *args, **kwargs: (
                observed.append(config) or {"status": "synthetic_image_received"}))
        self.assertEqual(receipt["provider"], "openai_chat")
        self.assertEqual(observed[0]["llm_expert_provider"], "openai_chat")

    def test_rejects_embedded_credentials_and_unexpected_response(self):
        with self.assertRaisesRegex(ValueError, "without embedded credentials"):
            run_smoke(
                provider="openai_chat", url="https://user:secret@example/v1",
                model="vision", key_env="LABELAGENT_TEST_KEY")
        with self.assertRaisesRegex(ValueError, "unexpected smoke response"):
            run_smoke(
                provider="openai_chat", url="https://api.example/v1",
                model="vision", key_env="LABELAGENT_TEST_KEY", execute=True,
                acknowledge_synthetic_upload=True,
                environ={"LABELAGENT_TEST_KEY": "secret"},
                chat_fn=lambda *args, **kwargs: {"status": "wrong"})

    def test_rejects_control_character_credential_without_network_or_secret_echo(self):
        calls = []
        malformed = "private-secret-value\r"
        with self.assertRaisesRegex(ValueError, "控制字符") as captured:
            run_smoke(
                provider="openai_chat", url="https://api.example/v1",
                model="vision", key_env="LABELAGENT_TEST_KEY", execute=True,
                acknowledge_synthetic_upload=True,
                environ={"LABELAGENT_TEST_KEY": malformed},
                chat_fn=lambda *args, **kwargs: calls.append(args))
        self.assertEqual(calls, [])
        self.assertNotIn("private-secret-value", str(captured.exception))


if __name__ == "__main__":
    unittest.main()
