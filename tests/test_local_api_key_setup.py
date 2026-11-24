import unittest

from tools.local_api_key_setup import PROVIDERS, page


class LocalApiKeySetupTests(unittest.TestCase):
    def test_page_is_local_only_and_never_contains_a_key_value(self):
        rendered = page("safe-token").decode("utf-8")
        self.assertIn("127.0.0.1", rendered)
        self.assertIn('type="password"', rendered)
        self.assertIn('value="safe-token"', rendered)
        self.assertNotIn("sk-test-secret", rendered)
        self.assertNotIn("api.openai.com", rendered)

    def test_supported_provider_environment_names_are_explicit(self):
        self.assertEqual(PROVIDERS["openai"][1], "LABELAGENT_OPENAI_API_KEY")
        self.assertEqual(PROVIDERS["qwen"][1], "LABELAGENT_QWEN_API_KEY")
        self.assertEqual(PROVIDERS["custom"][1], "LABELAGENT_EXPERT_API_KEY")


if __name__ == "__main__":
    unittest.main()
