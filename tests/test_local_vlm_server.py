import base64
import importlib.util
import io
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from PIL import Image

SPEC = importlib.util.spec_from_file_location("local_vlm_server", Path(__file__).resolve().parents[1] / "tools/local_vlm_server.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ContractBackend:
    model_id = "test-contract-model"
    model_loaded = True
    backend_name = "contract_test_double"

    def generate(self, messages, images, max_tokens, response_schema=None):
        self.generate_calls = getattr(self, "generate_calls", 0) + 1
        self.received_schema = response_schema
        self.sizes = [image.size for image in images]
        return '{"protocol_test":true}', 20, 5


class BlockingBackend(ContractBackend):
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def generate(self, messages, images, max_tokens, response_schema=None):
        self.started.set()
        if not self.release.wait(2):
            raise RuntimeError("test backend release timed out")
        return super().generate(messages, images, max_tokens, response_schema)


def messages(count=1):
    parts = [{"type": "text", "text": "Return JSON"}]
    for index in range(count):
        output = io.BytesIO()
        Image.new("RGB", (10+index, 8)).save(output, "PNG")
        parts.append({"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()}})
    return [{"role": "user", "content": parts}]


class LocalVLMContractTests(unittest.TestCase):
    def test_structured_decoder_tokenizer_cache_builds_exactly_once(self):
        backend = SimpleNamespace(processor=SimpleNamespace(
            tokenizer=object()))
        with patch.object(MODULE, "build_format_tokenizer_data",
                          return_value="cached") as build:
            self.assertEqual(MODULE.ensure_format_tokenizer_data(backend), "cached")
            self.assertEqual(MODULE.ensure_format_tokenizer_data(backend), "cached")
        build.assert_called_once_with(backend.processor.tokenizer)

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

    def test_exact_replay_uses_bounded_response_cache(self):
        body = {"model": self.backend.model_id, "messages": messages()}
        first = self.client.post("/v1/chat/completions", json=body)
        with patch.object(MODULE, "decode_messages",
                          side_effect=AssertionError("cache hit decoded media again")):
            second = self.client.post("/v1/chat/completions", json=body)
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(self.backend.generate_calls, 1)
        self.assertFalse(first.json["local_vlm"]["cache_hit"])
        self.assertTrue(second.json["local_vlm"]["cache_hit"])
        self.assertEqual(second.json["local_vlm"]["response_cache_entries"], 1)
        self.assertEqual(first.json["choices"], second.json["choices"])
        health = self.client.get("/health").json
        self.assertEqual(health["response_cache_capacity"], 32)
        self.assertEqual(health["response_cache_entries"], 1)

    def test_cache_key_includes_generation_contract(self):
        body = {"model": self.backend.model_id, "messages": messages()}
        self.client.post("/v1/chat/completions", json=body)
        changed = self.client.post("/v1/chat/completions", json={**body, "max_tokens": 17})
        self.assertEqual(changed.status_code, 200)
        self.assertEqual(self.backend.generate_calls, 2)
        self.assertFalse(changed.json["local_vlm"]["cache_hit"])

    def test_identical_inflight_requests_share_one_generation(self):
        backend = BlockingBackend()
        app = MODULE.create_app(backend, coalesce_timeout_seconds=2)
        body = {"model": backend.model_id, "messages": messages()}
        replies = []
        def post():
            with app.test_client() as client:
                replies.append(client.post("/v1/chat/completions", json=body))
        first = threading.Thread(target=post)
        second = threading.Thread(target=post)
        first.start()
        self.assertTrue(backend.started.wait(1))
        second.start()
        backend.release.set()
        first.join(2)
        second.join(2)
        self.assertEqual([reply.status_code for reply in replies], [200, 200])
        self.assertEqual(backend.generate_calls, 1)
        metadata = [reply.json["local_vlm"] for reply in replies]
        self.assertEqual(sum(item["cache_hit"] for item in metadata), 1)
        self.assertEqual(sum(item.get("coalesced", False) for item in metadata), 1)

    def test_different_inflight_request_keeps_busy_backpressure(self):
        backend = BlockingBackend()
        app = MODULE.create_app(backend, coalesce_timeout_seconds=2)
        body = {"model": backend.model_id, "messages": messages()}
        holder = []
        def post_first():
            with app.test_client() as client:
                holder.append(client.post("/v1/chat/completions", json=body))
        first = threading.Thread(target=post_first)
        first.start()
        self.assertTrue(backend.started.wait(1))
        with app.test_client() as client:
            different = client.post("/v1/chat/completions",
                                    json={**body, "max_tokens": 17})
        self.assertEqual(different.status_code, 503)
        self.assertEqual(different.json["error"]["message"], "Local VLM is busy")
        backend.release.set()
        first.join(2)
        self.assertEqual(holder[0].status_code, 200)
        self.assertEqual(backend.generate_calls, 1)


if __name__ == "__main__":
    unittest.main()
