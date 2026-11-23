import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from urllib.parse import parse_qs, urlsplit

from PIL import Image

from tools.snapshot_workbench_frames import snapshot


class Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.payload


class Workbench:
    def __init__(self, *, drift=False, busy=False):
        self.path = str(Path(tempfile.gettempdir(), "source.mcap").resolve())
        self.topic = "/camera"
        self.generation = 123
        self.requests = []
        self.drift = drift
        self.busy = busy
        self.bootstrap_reads = 0
        target = io.BytesIO()
        Image.new("RGB", (20, 12), (120, 130, 140)).save(target, "JPEG")
        self.image = target.getvalue()

    def __call__(self, request, timeout=0):
        self.requests.append(request)
        if request.get_method() != "GET":
            raise AssertionError("snapshot attempted a non-GET request")
        parsed = urlsplit(request.full_url)
        if parsed.path == "/api/bootstrap":
            self.bootstrap_reads += 1
            generation = self.generation + (
                1 if self.drift and self.bootstrap_reads > 1 else 0)
            value = {"loaded": True, "path": self.path, "topic": self.topic,
                     "file_activation_generation": generation,
                     "catalog": {self.topic: {"count": 10}}}
            return Response(json.dumps(value).encode())
        if parsed.path == "/api/run/status":
            query = parse_qs(parsed.query)
            if query.get("path") != [self.path] or query.get("topic") != [self.topic]:
                raise AssertionError("status scope mismatch")
            value = {"status": "running" if self.busy else "idle",
                     "selected_file": self.path,
                     "selected_topic": self.topic,
                     "selected_file_activation_generation": self.generation}
            return Response(json.dumps(value).encode())
        if parsed.path in {"/api/frame/2", "/api/frame/7"}:
            return Response(self.image)
        raise AssertionError(f"unexpected URL {request.full_url}")


class SnapshotWorkbenchFramesTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_snapshots_hash_bound_images_using_only_get(self):
        workbench = Workbench()
        output = self.root / "snapshot"

        result = snapshot(
            "http://example.test", [2, 7], output, opener=workbench)

        self.assertEqual(result["image_count"], 2)
        self.assertTrue(all(item.get_method() == "GET"
                            for item in workbench.requests))
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertFalse(manifest["truth_read"])
        self.assertFalse(manifest["workbench_write"])
        self.assertEqual(manifest["requested_frames"], [2, 7])
        for item in manifest["images"]:
            payload = (output / item["image"]).read_bytes()
            self.assertEqual(hashlib.sha256(payload).hexdigest(), item["sha256"])

    def test_busy_workbench_fails_before_output(self):
        output = self.root / "busy"
        with self.assertRaisesRegex(ValueError, "not idle"):
            snapshot("http://example.test", [2], output,
                     opener=Workbench(busy=True))
        self.assertFalse(output.exists())

    def test_scope_drift_fails_without_output_or_staging(self):
        output = self.root / "drift"
        with self.assertRaisesRegex(ValueError, "scope changed"):
            snapshot("http://example.test", [2], output,
                     opener=Workbench(drift=True))
        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.glob(".snapshot-*")), [])

    def test_rejects_duplicate_or_existing_output(self):
        with self.assertRaisesRegex(ValueError, "duplicates"):
            snapshot("http://example.test", [2, 2], self.root / "duplicate",
                     opener=Workbench())
        existing = self.root / "existing"
        existing.mkdir()
        with self.assertRaises(FileExistsError):
            snapshot("http://example.test", [2], existing,
                     opener=Workbench())


if __name__ == "__main__":
    unittest.main()
