import base64
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
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

    def generate_batch(self, requests, max_tokens, response_schema=None):
        self.generate_batch_calls = getattr(self, "generate_batch_calls", 0) + 1
        self.received_batch_schema = response_schema
        self.batch_sizes = [[image.size for image in images]
                            for _, images in requests]
        self.last_inference_stats = {"batch_size": len(requests)}
        return [('{"protocol_test":true}', 20 + index, 5)
                for index in range(len(requests))]


class BlockingBackend(ContractBackend):
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def generate(self, messages, images, max_tokens, response_schema=None):
        self.started.set()
        if not self.release.wait(2):
            raise RuntimeError("test backend release timed out")
        return super().generate(messages, images, max_tokens, response_schema)


class ThreadAffinityBackend(ContractBackend):
    startup_prewarm_enabled = True
    startup_prewarm = None
    model_loaded = False

    def __init__(self):
        self.inference_threads = []

    def _run_startup_prewarm(self):
        self.inference_threads.append(threading.get_ident())
        return {"performed": True}

    def generate(self, messages, images, max_tokens, response_schema=None):
        self.inference_threads.append(threading.get_ident())
        return super().generate(messages, images, max_tokens, response_schema)


def messages(count=1):
    parts = [{"type": "text", "text": "Return JSON"}]
    for index in range(count):
        output = io.BytesIO()
        Image.new("RGB", (10+index, 8)).save(output, "PNG")
        parts.append({"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()}})
    return [{"role": "user", "content": parts}]


def model_asset_fixture(base):
    model_dir = Path(base) / "model"
    model_dir.mkdir()
    contents = {
        "config.json": b"{}\n",
        "generation_config.json": b"{}\n",
        "model.safetensors.index.json": b"{}\n",
        "preprocessor_config.json": b"{}\n",
        "tokenizer_config.json": b"{}\n",
        "tokenizer.json": b"{}\n",
        "model-00001-of-00001.safetensors": b"test-weights",
    }
    records = []
    for name, content in contents.items():
        path = model_dir / name
        path.write_bytes(content)
        stat = path.stat()
        records.append({"name": name, "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                        "sha256": hashlib.sha256(content).hexdigest()})
    manifest = Path(base) / "model-assets.json"
    manifest.write_text(json.dumps({
        "schema_version": "labelagent.model-assets.v1",
        "files": records,
    }), encoding="utf-8")
    return model_dir, manifest


class LocalVLMContractTests(unittest.TestCase):
    def test_model_asset_manifest_binds_complete_runtime_set(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir, manifest = model_asset_fixture(directory)
            fingerprint = MODULE.validate_model_asset_manifest(
                model_dir, manifest)
            self.assertEqual(len(fingerprint), 64)

            (model_dir / "tokenizer.json").write_bytes(b"changed\n")
            with self.assertRaisesRegex(ValueError, "metadata differs"):
                MODULE.validate_model_asset_manifest(model_dir, manifest)

    def test_model_asset_manifest_rejects_partial_and_unsafe_records(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir, manifest = model_asset_fixture(directory)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["files"].pop()
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "complete runtime file set"):
                MODULE.validate_model_asset_manifest(model_dir, manifest)

            payload["files"].append({
                "name": "../outside.json", "size": 0, "mtime_ns": 0,
                "sha256": "0" * 64})
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid model asset record"):
                MODULE.validate_model_asset_manifest(model_dir, manifest)

    def test_persistent_response_cache_survives_restart_without_request_data(self):
        with tempfile.TemporaryDirectory() as directory:
            body = {"model": ContractBackend.model_id,
                    "messages": messages(), "max_tokens": 23}
            first_backend = ContractBackend()
            first_backend.model_assets_sha256 = "a" * 64
            first_app = MODULE.create_app(
                first_backend, persistent_response_cache_dir=directory)
            first = first_app.test_client().post(
                "/v1/chat/completions", json=body)
            self.assertEqual(first.status_code, 200)
            self.assertEqual(first_backend.generate_calls, 1)

            records = list(Path(directory).glob("*/*.json"))
            self.assertEqual(len(records), 1)
            persisted = records[0].read_text(encoding="utf-8")
            self.assertNotIn("data:image", persisted)
            self.assertNotIn('"messages"', persisted)
            self.assertNotIn("Return JSON", persisted)

            restarted_backend = ContractBackend()
            restarted_backend.model_assets_sha256 = "a" * 64
            restarted_app = MODULE.create_app(
                restarted_backend, persistent_response_cache_dir=directory)
            restarted = restarted_app.test_client().post(
                "/v1/chat/completions", json=body)
            self.assertEqual(restarted.status_code, 200)
            self.assertFalse(hasattr(restarted_backend, "generate_calls"))
            self.assertTrue(restarted.json["local_vlm"]["cache_hit"])
            self.assertTrue(
                restarted.json["local_vlm"]["persistent_cache_hit"])
            health = restarted_app.test_client().get("/health").json
            self.assertEqual(
                health["persistent_response_cache"]["loaded_at_start"], 1)
            self.assertEqual(
                health["persistent_response_cache"]["invalid_files"], 0)

            changed_backend = ContractBackend()
            changed_backend.model_assets_sha256 = "b" * 64
            changed_app = MODULE.create_app(
                changed_backend, persistent_response_cache_dir=directory)
            changed = changed_app.test_client().post(
                "/v1/chat/completions", json=body)
            self.assertEqual(changed.status_code, 200)
            self.assertEqual(changed_backend.generate_calls, 1)
            self.assertFalse(changed.json["local_vlm"]["cache_hit"])
            self.assertEqual(len([path for path in Path(directory).iterdir()
                                  if path.is_dir()]), 2)

    def test_persistent_response_cache_supports_deep_windows_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            deep = Path(directory)
            # Keep the final hashed cache path valid while making the old
            # duplicated-hash temporary name exceed a typical MAX_PATH.
            while len(str(deep)) < 100:
                deep /= "deep-checkout"
            deep.mkdir(parents=True, exist_ok=True)
            body = {"model": ContractBackend.model_id,
                    "messages": messages(), "max_tokens": 23}
            backend = ContractBackend()
            backend.model_assets_sha256 = "b" * 64
            app = MODULE.create_app(
                backend, persistent_response_cache_dir=deep)
            response = app.test_client().post(
                "/v1/chat/completions", json=body)
            self.assertEqual(response.status_code, 200)
            records = list(deep.glob("*/*.json"))
            self.assertEqual(len(records), 1)
            self.assertLessEqual(len(records[0].name), 69)

    def test_persistent_cache_write_failure_never_fails_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = ContractBackend()
            backend.model_assets_sha256 = "c" * 64
            app = MODULE.create_app(
                backend, persistent_response_cache_dir=directory)
            body = {"model": backend.model_id, "messages": messages()}
            with patch.object(MODULE.os, "replace",
                              side_effect=OSError("read-only cache")):
                first = app.test_client().post(
                    "/v1/chat/completions", json=body)
            second = app.test_client().post(
                "/v1/chat/completions", json=body)
            self.assertEqual(first.status_code, 200)
            self.assertEqual(second.status_code, 200)
            self.assertEqual(backend.generate_calls, 1)
            self.assertTrue(second.json["local_vlm"]["cache_hit"])
            health = app.test_client().get("/health").json
            self.assertEqual(
                health["persistent_response_cache"]["write_failures"], 1)

    def test_prewarm_and_requests_share_one_inference_thread(self):
        backend = ThreadAffinityBackend()
        app = MODULE.create_app(backend)
        client = app.test_client()
        first = client.post("/v1/chat/completions", json={
            "model": backend.model_id, "messages": messages(),
            "max_tokens": 10})
        second = client.post("/v1/chat/completions", json={
            "model": backend.model_id, "messages": messages(),
            "max_tokens": 11})

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertTrue(backend.model_loaded)
        self.assertEqual(backend.startup_prewarm, {"performed": True})
        self.assertEqual(len(backend.inference_threads), 3)
        self.assertEqual(len(set(backend.inference_threads)), 1)

    def test_generation_cache_implementation_is_explicit_and_bounded(self):
        self.assertIsNone(MODULE.checked_generation_cache_implementation(None))
        self.assertEqual(MODULE.checked_generation_cache_implementation("dynamic"), "dynamic")
        self.assertEqual(MODULE.checked_generation_cache_implementation("static"), "static")
        for value in ("offloaded", "quantized", "", True):
            with self.assertRaises(ValueError):
                MODULE.checked_generation_cache_implementation(value)

    def test_static_cache_is_guarded_for_multi_image_agent_review(self):
        self.assertEqual(
            MODULE.request_generation_cache_implementation("static", 1),
            ("static", None))
        self.assertEqual(
            MODULE.request_generation_cache_implementation("static", 3),
            ("dynamic", "multi_image_static_cache_guard"))
        self.assertEqual(
            MODULE.request_generation_cache_implementation("dynamic", 3),
            ("dynamic", None))
        self.assertEqual(
            MODULE.request_generation_cache_implementation(None, 2),
            (None, None))

    def test_startup_prewarm_is_restricted_to_exclusive_bf16_static_mode(self):
        self.assertFalse(MODULE.checked_startup_prewarm(
            False, None, "none", None))
        self.assertTrue(MODULE.checked_startup_prewarm(
            True, "static", "none", None))
        for arguments in ((True, None, "none", None),
                          (True, "dynamic", "none", None),
                          (True, "static", "nf4", None),
                          (True, "static", "none", .35),
                          ("yes", "static", "none", None)):
            with self.assertRaises(ValueError):
                MODULE.checked_startup_prewarm(*arguments)

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

    def test_independent_batch_keeps_request_and_image_boundaries(self):
        schema = {"type": "object", "properties": {
            "protocol_test": {"type": "boolean"}}}
        item = {"model": self.backend.model_id, "messages": messages(2),
                "max_tokens": 32,
                "response_format": {"type": "json_schema",
                    "json_schema": {"schema": schema}}}
        result = self.client.post(
            "/v1/chat/batch_completions",
            json={"requests": [item, {**item, "messages": messages(3)}]})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(self.backend.generate_batch_calls, 1)
        self.assertEqual(self.backend.batch_sizes,
                         [[(10, 8), (11, 8)],
                          [(10, 8), (11, 8), (12, 8)]])
        self.assertEqual(self.backend.received_batch_schema, schema)
        self.assertEqual(result.json["local_vlm"]["batch_size"], 2)
        self.assertEqual(len(result.json["results"]), 2)

    def test_independent_batch_reuses_exact_single_request_cache(self):
        first = {"model": self.backend.model_id, "messages": messages(2),
                 "max_tokens": 32}
        second = {"model": self.backend.model_id, "messages": messages(3),
                  "max_tokens": 32}
        self.assertEqual(self.client.post(
            "/v1/chat/completions", json=first).status_code, 200)
        self.assertEqual(self.client.post(
            "/v1/chat/completions", json=second).status_code, 200)
        result = self.client.post(
            "/v1/chat/batch_completions", json={"requests": [first, second]})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(getattr(self.backend, "generate_batch_calls", 0), 0)
        self.assertEqual(self.backend.generate_calls, 2)
        self.assertTrue(result.json["local_vlm"]["cache_hit"])
        self.assertEqual(result.json["local_vlm"]["cache_hit_count"], 2)
        self.assertEqual(result.json["local_vlm"]["generated_count"], 0)
        self.assertTrue(all(row["local_vlm"]["cache_hit"]
                            for row in result.json["results"]))

    def test_independent_batch_generates_only_one_uncached_request(self):
        cached = {"model": self.backend.model_id, "messages": messages(2),
                  "max_tokens": 32}
        missing = {"model": self.backend.model_id, "messages": messages(3),
                   "max_tokens": 32}
        self.assertEqual(self.client.post(
            "/v1/chat/completions", json=cached).status_code, 200)
        result = self.client.post(
            "/v1/chat/batch_completions", json={"requests": [cached, missing]})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(getattr(self.backend, "generate_batch_calls", 0), 0)
        self.assertEqual(self.backend.generate_calls, 2)
        self.assertEqual(result.json["local_vlm"]["cache_hit_count"], 1)
        self.assertEqual(result.json["local_vlm"]["generated_count"], 1)
        replay = self.client.post(
            "/v1/chat/batch_completions", json={"requests": [cached, missing]})
        self.assertEqual(replay.status_code, 200)
        self.assertTrue(replay.json["local_vlm"]["cache_hit"])
        self.assertEqual(self.backend.generate_calls, 2)

    def test_independent_batch_populates_single_request_cache(self):
        first = {"model": self.backend.model_id, "messages": messages(2),
                 "max_tokens": 32}
        second = {"model": self.backend.model_id, "messages": messages(3),
                  "max_tokens": 32}
        result = self.client.post(
            "/v1/chat/batch_completions", json={"requests": [first, second]})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(self.backend.generate_batch_calls, 1)
        self.assertEqual(result.json["local_vlm"]["generated_count"], 2)
        self.assertEqual(self.client.post(
            "/v1/chat/completions", json=first).status_code, 200)
        self.assertEqual(self.client.post(
            "/v1/chat/completions", json=second).status_code, 200)
        self.assertEqual(getattr(self.backend, "generate_calls", 0), 0)

    def test_batch_rejects_mixed_contracts_and_unsafe_sizes(self):
        base = {"model": self.backend.model_id, "messages": messages(),
                "max_tokens": 32}
        bad_bodies = [
            {"requests": [base]},
            {"requests": [base] * 4},
            {"requests": [base, {**base, "max_tokens": 33}]},
            {"requests": [base, {**base, "image_token_budget": 128}]},
        ]
        for body in bad_bodies:
            with self.subTest(body=body):
                self.assertEqual(self.client.post(
                    "/v1/chat/batch_completions", json=body).status_code, 400)

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
        self.assertEqual(health["response_cache_capacity"], 64)
        self.assertEqual(health["response_cache_entries"], 1)
        self.assertEqual(health["response_cache_stats"], {
            "hits": 1, "misses": 1, "stores": 1, "evictions": 0,
            "coalesced": 0, "coalesce_timeouts": 0,
            "coalesce_reuse_failures": 0, "busy_rejections": 0,
        })

    def test_response_cache_is_bounded_lru_and_refreshes_exact_hits(self):
        backend = ContractBackend()
        client = MODULE.create_app(backend, response_cache_entries=2).test_client()
        bodies = [{"model": backend.model_id, "messages": messages(),
                   "max_tokens": value} for value in (10, 11, 12)]

        self.assertFalse(client.post(
            "/v1/chat/completions", json=bodies[0]).json["local_vlm"]["cache_hit"])
        self.assertFalse(client.post(
            "/v1/chat/completions", json=bodies[1]).json["local_vlm"]["cache_hit"])
        self.assertTrue(client.post(
            "/v1/chat/completions", json=bodies[0]).json["local_vlm"]["cache_hit"])
        self.assertFalse(client.post(
            "/v1/chat/completions", json=bodies[2]).json["local_vlm"]["cache_hit"])
        self.assertTrue(client.post(
            "/v1/chat/completions", json=bodies[0]).json["local_vlm"]["cache_hit"])
        self.assertFalse(client.post(
            "/v1/chat/completions", json=bodies[1]).json["local_vlm"]["cache_hit"])

        self.assertEqual(backend.generate_calls, 4)
        health = client.get("/health").json
        self.assertEqual(health["response_cache_capacity"], 2)
        self.assertEqual(health["response_cache_entries"], 2)
        self.assertEqual(health["response_cache_stats"], {
            "hits": 2, "misses": 4, "stores": 4, "evictions": 2,
            "coalesced": 0, "coalesce_timeouts": 0,
            "coalesce_reuse_failures": 0, "busy_rejections": 0,
        })

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
        health = app.test_client().get("/health").json
        self.assertEqual(health["response_cache_stats"], {
            "hits": 0, "misses": 2, "stores": 1, "evictions": 0,
            "coalesced": 1, "coalesce_timeouts": 0,
            "coalesce_reuse_failures": 0, "busy_rejections": 0,
        })

    def test_different_inflight_request_waits_in_server_queue(self):
        backend = BlockingBackend()
        app = MODULE.create_app(
            backend, coalesce_timeout_seconds=2,
            inference_queue_timeout_seconds=2)
        body = {"model": backend.model_id, "messages": messages()}
        replies = []
        def post(payload):
            with app.test_client() as client:
                replies.append(client.post("/v1/chat/completions", json=payload))
        first = threading.Thread(target=post, args=(body,))
        first.start()
        self.assertTrue(backend.started.wait(1))
        second = threading.Thread(
            target=post, args=({**body, "max_tokens": 17},))
        second.start()
        for _ in range(100):
            queue = app.test_client().get("/health").json["inference_queue"]
            if queue["waiting"] == 1:
                break
            threading.Event().wait(.01)
        self.assertEqual(queue["waiting"], 1)
        threading.Event().wait(.02)
        backend.release.set()
        first.join(2)
        second.join(2)
        self.assertEqual([reply.status_code for reply in replies], [200, 200])
        self.assertEqual(backend.generate_calls, 2)
        self.assertEqual(
            sum(reply.json["local_vlm"]["queue_wait_seconds"] > 0
                for reply in replies), 1)
        health = app.test_client().get("/health").json
        self.assertEqual(health["response_cache_stats"]["misses"], 2)
        self.assertEqual(health["response_cache_stats"]["busy_rejections"], 0)
        self.assertEqual(health["response_cache_stats"]["stores"], 2)
        self.assertEqual(health["inference_queue"]["waiting"], 0)
        self.assertEqual(health["inference_queue"]["total_waits"], 1)
        self.assertEqual(health["inference_queue"]["timeouts"], 0)
        self.assertEqual(health["inference_queue"]["max_waiting"], 1)


if __name__ == "__main__":
    unittest.main()
