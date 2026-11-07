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


if __name__ == "__main__":
    unittest.main()
