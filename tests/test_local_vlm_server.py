import base64
import importlib.util
import io
from pathlib import Path
import unittest
from PIL import Image

SPEC = importlib.util.spec_from_file_location("local_vlm_server", Path(__file__).resolve().parents[1] / "tools/local_vlm_server.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ContractBackend:
    model_id = "test-contract-model"
    model_loaded = True
    backend_name = "contract_test_double"

    def generate(self, messages, images, max_tokens, response_schema=None):
        self.received_schema = response_schema
        self.sizes = [image.size for image in images]
        return '{"protocol_test":true}', 20, 5


def messages(count=1):
    parts = [{"type": "text", "text": "Return JSON"}]
    for index in range(count):
        output = io.BytesIO()
        Image.new("RGB", (10+index, 8)).save(output, "PNG")
        parts.append({"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()}})
    return [{"role": "user", "content": parts}]


class LocalVLMContractTests(unittest.TestCase):
    def test_structured_format_is_forwarded_to_backend(self):
        schema = {"type": "object", "properties": {"status": {"enum": ["done", "continue"]}}}
        result = self.client.post("/v1/chat/completions", json={"model": self.backend.model_id,
            "messages": messages(), "response_format": {"type": "json_schema", "json_schema": {"schema": schema}}})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(self.backend.received_schema, schema)

    def test_unknown_structured_format_is_rejected_instead_of_ignored(self):
        result = self.client.post("/v1/chat/completions", json={"model": self.backend.model_id,
            "messages": messages(), "response_format": {"type": "unsupported"}})
        self.assertEqual(result.status_code, 400)

    def setUp(self):
        self.backend = ContractBackend()
        self.client = MODULE.create_app(self.backend).test_client()

    def test_five_images_keep_order_and_api_has_chat_completion_shape(self):
        result = self.client.post("/v1/chat/completions", json={"model": self.backend.model_id, "messages": messages(5)})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(self.backend.sizes, [(10+i, 8) for i in range(5)])
        self.assertEqual(result.json["local_vlm"]["backend"], "contract_test_double")
        self.assertEqual(result.json["usage"]["total_tokens"], 25)
        self.assertEqual(result.json["choices"][0]["message"]["content"], '{"protocol_test":true}')

    def test_remote_image_fetching_and_invalid_token_budget_are_rejected(self):
        data = messages()
        data[0]["content"][1]["image_url"]["url"] = "https://example.test/private.png"
        for body in ({"messages": data}, {"messages": messages(), "max_tokens": 999999},
                     {"messages": messages(), "max_tokens": True}):
            result = self.client.post("/v1/chat/completions", json={"model": self.backend.model_id, **body})
            self.assertEqual(result.status_code, 400)

    def test_too_many_images_and_wrong_model_are_rejected(self):
        self.assertEqual(self.client.post("/v1/chat/completions", json={"model": self.backend.model_id, "messages": messages(9)}).status_code, 400)
        self.assertEqual(self.client.post("/v1/chat/completions", json={"model": "other", "messages": messages()}).status_code, 400)

    def test_output_limit_and_invalid_media_are_reported_without_false_success(self):
        result = self.client.post("/v1/chat/completions", json={"model": self.backend.model_id, "messages": messages(), "max_tokens": 5})
        self.assertEqual(result.json["choices"][0]["finish_reason"], "length")
        data = messages()
        data[0]["content"][1]["image_url"] = "bad"
        self.assertEqual(self.client.post("/v1/chat/completions", json={"model": self.backend.model_id, "messages": data}).status_code, 400)


if __name__ == "__main__":
    unittest.main()
