import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from urllib.parse import parse_qs, urlsplit

from PIL import Image

from source.agent.candidate_class_audit import candidate_fingerprint
from source.agent.candidate_comparison import read_packet
from tools import build_pallet_boundary_comparison_pack as builder


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.payload


def image_bytes():
    target = io.BytesIO()
    Image.new("RGB", (64, 64), (125, 125, 125)).save(target, "PNG")
    return target.getvalue()


def candidate(changed=True):
    value = {
        "id": 712, "label": "pallet", "score": .9,
        "points": [[5, 5], [55, 5], [55, 55], [5, 55]], "holes": [],
    }
    if changed:
        value["holes"] = [[[20, 20], [25, 20], [25, 25], [20, 25]]]
    return value


class FakeWorkbench:
    def __init__(self, candidates):
        self.requests = []
        self.path = str(Path(tempfile.gettempdir(), "generic-source.mcap").resolve())
        self.topic = "/generic/camera"
        self.generation = 123456
        self.frame = 7
        self.image = image_bytes()
        self.candidates = candidates
        self.binding = {
            "candidate_artifact_sha256": builder._candidate_binding(candidates),
            "candidates": [{
                "candidate_id": item["id"],
                "candidate_fingerprint": candidate_fingerprint(item),
            } for item in candidates],
        }

    def __call__(self, request, timeout=0):
        self.requests.append(request)
        self.assert_get(request)
        parsed = urlsplit(request.full_url)
        if parsed.path == "/api/bootstrap":
            value = {"loaded": True, "path": self.path, "topic": self.topic,
                     "file_activation_generation": self.generation,
                     "catalog": {self.topic: {"count": 20}}}
            return FakeResponse(json.dumps(value).encode())
        if parsed.path == f"/api/frame_state/{self.frame}":
            query = parse_qs(parsed.query)
            if (query.get("path") != [self.path]
                    or query.get("topic") != [self.topic]
                    or query.get("file_activation_generation") !=
                    [str(self.generation)]):
                raise AssertionError("scope query mismatch")
            value = {
                "path": self.path, "topic": self.topic,
                "file_activation_generation": self.generation,
                "frame": self.frame,
                "candidates": {str(self.frame): self.candidates},
                "candidate_review_bindings": {str(self.frame): self.binding},
                # Truth can be present in the server response, but the builder
                # intentionally never reads or copies it.
                "truth": {str(self.frame): [{"label": "ignored-human-truth"}]},
            }
            return FakeResponse(json.dumps(value).encode())
        if parsed.path == f"/api/frame/{self.frame}":
            return FakeResponse(self.image)
        raise AssertionError(f"unexpected URL {request.full_url}")

    @staticmethod
    def assert_get(request):
        if request.get_method() != "GET":
            raise AssertionError("builder attempted a non-GET request")


class BuildPalletBoundaryComparisonPackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_builds_generic_hash_bound_pack_using_only_get(self):
        workbench = FakeWorkbench([candidate(True), {
            "id": 99, "label": "person", "score": .8,
            "points": [[1, 1], [3, 1], [3, 3], [1, 3]], "holes": [],
        }])
        output = self.root / "run-a" / "generic-pack"
        result = builder.build_pack(
            "http://example.test:9999", [7], output, opener=workbench)
        self.assertEqual(result["comparison_count"], 1)
        self.assertEqual(result["server_methods"], ["GET"])
        self.assertTrue(all(item.get_method() == "GET" for item in workbench.requests))
        self.assertEqual(sum(urlsplit(item.full_url).path == "/api/bootstrap"
                             for item in workbench.requests), 2)
        self.assertEqual(sum(urlsplit(item.full_url).path == "/api/frame_state/7"
                             for item in workbench.requests), 2)
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["source"]["file_activation_generation"], 123456)
        self.assertNotIn("ignored-human-truth", json.dumps(manifest))
        self.assertFalse(manifest["human_truth"])
        row = manifest["rows"][0]
        self.assertEqual(row["frame"], 7)
        self.assertEqual(row["current_candidate_id"], 712)
        self.assertGreater(row["variants"][0]["metrics"]["changed_pixels"], 0)
        self.assertNotEqual(row["current"]["mask_sha256"],
                            row["variants"][0]["mask_sha256"])
        validation_root = self.root / "validate"
        target = validation_root / "candidate_comparison_packs" / "generic-pack"
        target.parent.mkdir(parents=True)
        import shutil
        shutil.copytree(output, target)
        packet = read_packet(validation_root, "generic-pack")
        self.assertEqual(packet["comparison_count"], 1)

    def test_no_geometry_change_fails_without_leaving_output(self):
        # Area below the cleanup threshold is deliberately unchanged.
        value = {"id": 4, "label": "pallet", "score": .7,
                 "points": [[1, 1], [10, 1], [10, 10], [1, 10]], "holes": []}
        workbench = FakeWorkbench([value])
        output = self.root / "unchanged" / "generic-pack"
        with self.assertRaisesRegex(ValueError, "没有几何确实变化"):
            builder.build_pack("http://example.test", [7], output,
                               opener=workbench)
        self.assertFalse(output.exists())
        self.assertEqual(list(output.parent.glob(".cc-*")), [])

    def test_two_runs_with_same_parameters_are_byte_reproducible(self):
        first = self.root / "first" / "same-pack"
        second = self.root / "second" / "same-pack"
        result_a = builder.build_pack(
            "http://example.test", [7], first,
            opener=FakeWorkbench([candidate(True)]))
        result_b = builder.build_pack(
            "http://example.test", [7], second,
            opener=FakeWorkbench([candidate(True)]))
        self.assertEqual(result_a["manifest_sha256"], result_b["manifest_sha256"])
        self.assertEqual(result_a["tree_sha256"], result_b["tree_sha256"])
        files_a = {path.relative_to(first).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in first.rglob("*") if path.is_file()}
        files_b = {path.relative_to(second).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in second.rglob("*") if path.is_file()}
        self.assertEqual(files_a, files_b)

    def test_rejects_duplicate_frames_and_existing_output(self):
        output = self.root / "pack"
        with self.assertRaisesRegex(ValueError, "不得重复"):
            builder.build_pack("http://example.test", [7, 7], output,
                               opener=FakeWorkbench([candidate(True)]))
        output.mkdir()
        with self.assertRaises(FileExistsError):
            builder.build_pack("http://example.test", [7], output,
                               opener=FakeWorkbench([candidate(True)]))

    def test_rejects_non_integer_pallet_id_before_writing_assets(self):
        workbench = FakeWorkbench([candidate(True)])
        value = candidate(True)
        value["id"] = "../escape"
        workbench.candidates = [value]
        workbench.binding = {
            "candidate_artifact_sha256": builder._candidate_binding([value]),
            "candidates": [{"candidate_id": value["id"],
                            "candidate_fingerprint": "0" * 64}],
        }
        output = self.root / "bad-id" / "generic-pack"
        with self.assertRaisesRegex(ValueError, "ID 必须是非负整数"):
            builder.build_pack("http://example.test", [7], output,
                               opener=workbench)
        self.assertFalse(output.exists())
        self.assertFalse((self.root / "escape-current-mask.png").exists())
        self.assertEqual(list(output.parent.glob(".cc-*")), [])


if __name__ == "__main__":
    unittest.main()
