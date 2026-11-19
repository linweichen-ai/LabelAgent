import io
import json
import unittest
from unittest.mock import patch
from source.agent import workflow
from source.agent.protocol_schema import review_schema


class SchemaTransportTests(unittest.TestCase):
    def test_compact_schema_and_output_budget_reach_the_model_request(self):
        schema = review_schema(["pallet"], [{"id": 7}], [])
        response = io.BytesIO(json.dumps({"choices": [{"message": {"content": '{"status":"done"}'}}]}).encode())
        config = {"llm_url": "http://127.0.0.1:18868/v1", "llm_model": "local-test", "llm_prompt_profile": "compact"}
        with patch.object(workflow.urllib.request, "urlopen", return_value=response) as call:
            workflow._chat(config, "review", b"private-image", response_schema=schema)
        body = json.loads(call.call_args.args[0].data)
        self.assertEqual(body["max_tokens"], 768)
        self.assertEqual(body["response_format"]["json_schema"]["schema"], schema)
        self.assertEqual(len(body["messages"][0]["content"]), 2)

    def test_unsatisfiable_structured_decoder_retries_once_with_json_validation(self):
        schema = review_schema(["pallet", "floor"], [{"id": 9}], [])
        failure = workflow.urllib.error.HTTPError(
            "http://127.0.0.1:18868/v1/chat/completions", 400,
            "Bad Request", None, io.BytesIO(json.dumps({"error": {"message":
                "`prefix_allowed_tokens_fn` returned an empty list; constraint is unsatisfiable"}}
            ).encode()))
        success = io.BytesIO(json.dumps({"choices": [{"finish_reason": "stop",
            "message": {"content": '{"status":"done"}'}}]}).encode())
        raw = {}
        config = {"llm_url": "http://127.0.0.1:18868/v1",
                  "llm_model": "local-test", "llm_prompt_profile": "compact"}

        with patch.object(workflow.urllib.request, "urlopen",
                          side_effect=[failure, success]) as call:
            result = workflow._chat(
                config, "review", b"private-image", raw=raw,
                response_schema=schema)

        self.assertEqual(result, {"status": "done"})
        self.assertEqual(call.call_count, 2)
        first = json.loads(call.call_args_list[0].args[0].data)
        second = json.loads(call.call_args_list[1].args[0].data)
        self.assertIn("response_format", first)
        self.assertNotIn("response_format", second)
        self.assertEqual(raw["structured_fallback_reason"],
                         "decoder_constraint_unsatisfiable")


if __name__ == "__main__":
    unittest.main()
