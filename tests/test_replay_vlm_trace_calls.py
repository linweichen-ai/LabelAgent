import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.replay_vlm_trace_calls import load_case, replay, sha256


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class VlmTraceReplayTests(unittest.TestCase):
    def make_case(self, root):
        image = b"\xff\xd8bounded-test-image"
        digest = sha256(image)
        inspection = root / "inspection"
        inspection.mkdir()
        (inspection / f"{digest}.jpg").write_bytes(image)
        debug = root / "model_debug.json"
        debug.write_text(json.dumps({"trace": [{
            "id": "call-1",
            "frame": 3,
            "purpose": "test",
            "input": {
                "instruction": "SECRET_PROMPT",
                "response_schema": {"type": "object"},
                "max_tokens": 12,
                "images": [{"sha256": digest}],
            },
        }]}), encoding="utf-8")
        return f"{debug}|call-1|{inspection}", image

    def test_load_case_binds_images_but_keeps_metadata_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            spec, image = self.make_case(Path(directory))
            request, metadata = load_case(spec)
        self.assertEqual(request["max_tokens"], 12)
        self.assertEqual(request["messages"][0]["content"][0]["text"], "SECRET_PROMPT")
        self.assertIn("data:image/jpeg;base64,", request["messages"][0]["content"][1]["image_url"]["url"])
        bounded = json.dumps(metadata)
        self.assertNotIn("SECRET_PROMPT", bounded)
        self.assertNotIn(image.decode("latin1"), bounded)
        self.assertEqual(metadata["image_count"], 1)

    def test_load_case_rejects_image_digest_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec, _ = self.make_case(root)
            image_path = next((root / "inspection").iterdir())
            image_path.write_bytes(b"\xff\xd8changed")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                load_case(spec)

    def test_replay_outputs_hashes_and_bounded_model_result(self):
        response = {
            "choices": [{"message": {"content": '{"status":"done"}'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "local_vlm": {"inference_stats": {"generation_seconds": 1.25}},
        }
        metadata = {"call_id": "call-1", "image_sha256": ["a" * 64]}
        with patch("tools.replay_vlm_trace_calls.urllib.request.urlopen",
                   return_value=FakeResponse(response)):
            result = replay("http://127.0.0.1:1", {"prompt": "SECRET_PROMPT"}, metadata, 1)
        self.assertEqual(result["parsed"], {"status": "done"})
        self.assertIsNone(result["parse_error"])
        self.assertEqual(result["local_vlm"]["inference_stats"]["generation_seconds"], 1.25)
        self.assertNotIn("SECRET_PROMPT", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
