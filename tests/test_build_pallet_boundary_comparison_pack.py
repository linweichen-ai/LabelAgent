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
    def __init__(self, candidates, *, unbound_frames=(),
                 drift_unbound_on_second_read=False):
        self.requests = []
        self.path = str(Path(tempfile.gettempdir(), "generic-source.mcap").resolve())
        self.topic = "/generic/camera"
        self.generation = 123456
        self.frame = 7
        self.image = image_bytes()
        self.candidates = candidates
        self.unbound_frames = set(unbound_frames)
        self.drift_unbound_on_second_read = drift_unbound_on_second_read
        self.unbound_reads = {}
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
        if parsed.path.startswith("/api/frame_state/"):
            frame = int(parsed.path.rsplit("/", 1)[1])
            query = parse_qs(parsed.query)
            if (query.get("path") != [self.path]
                    or query.get("topic") != [self.topic]
                    or query.get("file_activation_generation") !=
                    [str(self.generation)]):
                raise AssertionError("scope query mismatch")
            if frame in self.unbound_frames:
                self.unbound_reads[frame] = self.unbound_reads.get(frame, 0) + 1
                changed = (self.drift_unbound_on_second_read
                           and self.unbound_reads[frame] > 1)
                value = {
                    "path": self.path, "topic": self.topic,
                    "file_activation_generation": self.generation,
                    "frame": frame,
                    "candidates": ({str(frame): self.candidates}
                                   if changed else {}),
                    "candidate_review_bindings": ({str(frame): self.binding}
                                                  if changed else {}),
                }
                return FakeResponse(json.dumps(value).encode())
            if frame != self.frame:
                raise AssertionError(f"unexpected frame {frame}")
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

    def test_evidence_only_policy_can_emit_hash_bound_noop_replay(self):
        workbench = FakeWorkbench([candidate(False)])
        output = self.root / "v7" / "replay-pack"

        result = builder.build_pack(
            "http://example.test", [7], output, opener=workbench,
            geometry_policy="appearance_evidence_only",
            include_unchanged=True)

        self.assertEqual(result["comparison_count"], 1)
        self.assertEqual(result["changed_comparison_count"], 0)
        self.assertEqual(result["geometry_policy"],
                         "appearance_evidence_only")
        manifest = json.loads((output / "manifest.json").read_text(
            encoding="utf-8"))
        self.assertTrue(manifest["include_unchanged"])
        row = manifest["rows"][0]
        self.assertEqual(row["current"]["mask_sha256"],
                         row["variants"][0]["mask_sha256"])
        self.assertEqual(row["variants"][0]["metrics"]["changed_pixels"], 0)
        self.assertEqual(row["variants"][0]["metrics"][
            "appearance_evidence_used"], 0)
        self.assertEqual(row["variants"][0]["alternative_id"],
                         "pallet-boundary-cleanup-v7-evidence-only")
        packet_root = self.root / "packet"
        target = packet_root / "candidate_comparison_packs" / "replay-pack"
        target.parent.mkdir(parents=True)
        import shutil
        shutil.copytree(output, target)
        self.assertEqual(read_packet(packet_root, "replay-pack")[
            "comparison_count"], 1)

    def test_roi_evidence_policy_is_explicit_shadow_variant(self):
        output = self.root / "v8" / "roi-shadow-pack"
        result = builder.build_pack(
            "http://example.test", [7], output,
            opener=FakeWorkbench([candidate(False)]),
            geometry_policy="appearance_evidence_roi",
            include_unchanged=True)
        manifest = json.loads((output / "manifest.json").read_text(
            encoding="utf-8"))

        self.assertEqual(result["geometry_policy"],
                         "appearance_evidence_roi")
        self.assertEqual(manifest["geometry_policy"],
                         "appearance_evidence_roi")
        variant = manifest["rows"][0]["variants"][0]
        self.assertEqual(variant["alternative_id"],
                         "pallet-boundary-cleanup-v8-roi-evidence")

    def test_explicitly_skips_only_empty_frames_without_binding(self):
        workbench = FakeWorkbench([candidate(False)], unbound_frames={8})
        output = self.root / "sparse" / "replay-pack"

        result = builder.build_pack(
            "http://example.test", [7, 8], output, opener=workbench,
            geometry_policy="appearance_evidence_only",
            include_unchanged=True, skip_unbound_empty_frames=True)

        self.assertEqual(result["requested_frames"], [7, 8])
        self.assertEqual(result["skipped_unbound_empty_frames"], [8])
        manifest = json.loads((output / "manifest.json").read_text(
            encoding="utf-8"))
        self.assertEqual(manifest["skipped_unbound_empty_frames"], [8])
        self.assertEqual([row["frame"] for row in manifest["rows"]], [7])
        self.assertFalse(any("/api/frame/8" in item.full_url
                             for item in workbench.requests))

    def test_unbound_empty_frame_still_fails_closed_by_default(self):
        workbench = FakeWorkbench([candidate(False)], unbound_frames={8})
        output = self.root / "sparse-default" / "replay-pack"

        with self.assertRaisesRegex(ValueError, "frame 8 缺少服务端候选绑定"):
            builder.build_pack(
                "http://example.test", [7, 8], output, opener=workbench,
                geometry_policy="appearance_evidence_only",
                include_unchanged=True)

        self.assertFalse(output.exists())

    def test_skip_rejects_malformed_or_ambiguous_empty_state(self):
        scope = {"path": "C:\\data.mcap", "topic": "/camera",
                 "file_activation_generation": 10}
        base = {**scope, "frame": 8,
                "candidates": {"8": []},
                "candidate_review_bindings": {}}
        invalid = [
            ({key: value for key, value in base.items() if key != "candidates"},
             "候选字段"),
            ({**base, "candidates": []}, "候选字段"),
            ({**base, "candidates": {"7": []}}, "外来帧键"),
            ({**base, "candidates": {"8": {}}}, "候选列表"),
            ({key: value for key, value in base.items()
              if key != "candidate_review_bindings"}, "候选绑定字段"),
            ({**base, "candidate_review_bindings": {"8": None}},
             "候选绑定类型"),
            ({**base, "candidate_review_bindings": {"8": []}},
             "候选绑定类型"),
            ({**base,
              "candidates": {"8": [], "7": []},
              "candidate_review_bindings": {"8": {}, "7": {}}},
             "外来帧键"),
            ({**base,
              "candidates": {"8": [], "7": []},
              "candidate_review_bindings": {"8": {}}},
             "外来帧键"),
            ({**base,
              "candidates": {"8": []},
              "candidate_review_bindings": {"8": {}, "7": {}}},
             "外来帧键"),
        ]
        for value, message in invalid:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    builder._validate_frame_state(
                        value, frame=8, scope=scope,
                        allow_unbound_empty=True)

        skipped, skipped_binding = builder._validate_frame_state(
            {**base, "candidates": {}, "candidate_review_bindings": {}},
            frame=8, scope=scope, allow_unbound_empty=True)
        self.assertIsNone(skipped)
        self.assertIsNone(skipped_binding)

        nonempty = {**base, "candidates": {"8": [candidate(False)]}}
        with self.assertRaisesRegex(ValueError, "缺少服务端候选绑定"):
            builder._validate_frame_state(
                nonempty, frame=8, scope=scope, allow_unbound_empty=True)

    def test_skipped_frame_drift_fails_before_output(self):
        workbench = FakeWorkbench(
            [candidate(False)], unbound_frames={8},
            drift_unbound_on_second_read=True)
        output = self.root / "sparse-drift" / "replay-pack"

        with self.assertRaisesRegex(ValueError, "空候选状态.*发生变化"):
            builder.build_pack(
                "http://example.test", [7, 8], output, opener=workbench,
                geometry_policy="appearance_evidence_only",
                include_unchanged=True, skip_unbound_empty_frames=True)

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
