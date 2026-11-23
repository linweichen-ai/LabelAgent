import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from source.agent import server
from source.agent.candidate_comparison import read_packet


def png_bytes(color=(20, 40, 60)):
    target = io.BytesIO()
    Image.new("RGB", (24, 18), color).save(target, "PNG")
    return target.getvalue()


class FakeVideo:
    def __init__(self, path, payload):
        self.path = Path(path)
        self.payload = payload

    def frame(self, topic, index):
        if topic != "/camera-x" or index != 7:
            raise ValueError("unexpected frame")
        return self.payload


class CandidateComparisonPacketApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.packet = self.work / "candidate_comparison_packs" / "generic-pack"
        self.packet.mkdir(parents=True)
        self.source_path = str((self.work / "arbitrary-source.mcap").resolve())
        self.frame_payload = png_bytes()
        self.current_image = png_bytes((30, 60, 90))
        self.alt_image = png_bytes((90, 60, 30))
        self.current_mask = png_bytes((0, 0, 0))
        self.alt_mask = png_bytes((255, 255, 255))
        (self.packet / "current.png").write_bytes(self.current_image)
        (self.packet / "alternative.png").write_bytes(self.alt_image)
        (self.packet / "current-mask.png").write_bytes(self.current_mask)
        (self.packet / "alternative-mask.png").write_bytes(self.alt_mask)
        self.candidate = {
            "id": 991, "label": "arbitrary-class", "score": 0.7,
            "points": [[1, 1], [10, 1], [10, 8], [1, 8]], "holes": [],
        }
        binding = server.candidate_review_binding([self.candidate])
        self.value = {
            "schema_version": "labelagent.candidate-comparison-pack.v1",
            "candidate_only": True,
            "human_truth": False,
            "approved_for_training": False,
            "training_authorized": False,
            "automatic_approval": False,
            "assistant_decision": None,
            "human_decision": None,
            "source": {"path": self.source_path, "topic": "/camera-x",
                       "file_activation_generation": 42},
            "rows": [{
                "comparison_id": "any-frame-any-object",
                "frame": 7,
                "source_frame_sha256": hashlib.sha256(self.frame_payload).hexdigest(),
                "current_candidate_id": 991,
                "current_candidate_artifact_sha256": binding["candidate_artifact_sha256"],
                "current_candidate_fingerprint": binding["candidates"][0]["candidate_fingerprint"],
                "current": {"image": "current.png", "image_sha256": hashlib.sha256(self.current_image).hexdigest(),
                            "mask": "current-mask.png", "mask_sha256": hashlib.sha256(self.current_mask).hexdigest()},
                "current_pixel_count": 77,
                "current_metrics": {"iou": 1.0},
                "variants": [{
                    "alternative_id": "policy-alpha", "kind": "alternative",
                    "label": "alpha", "policy": "generic-policy",
                    "mask_sha256": hashlib.sha256(self.alt_mask).hexdigest(),
                    "mask": "alternative-mask.png", "pixel_count": 79,
                    "preview": {"image": "alternative.png", "image_sha256": hashlib.sha256(self.alt_image).hexdigest()},
                    "metrics": {"iou_to_current": 0.8},
                }],
            }],
        }
        self.manifest = self.packet / "manifest.json"
        self.write_manifest()
        self.state = {
            "video": FakeVideo(self.source_path, self.frame_payload),
            "topic": "/camera-x", "candidates": {"7": [copy.deepcopy(self.candidate)]},
        }
        self.patches = [patch.object(server, "WORK", self.work),
                        patch.object(server, "state", self.state),
                        patch.object(server, "file_activation_generation", 42)]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def write_manifest(self):
        self.manifest.write_text(json.dumps(self.value), encoding="utf-8")

    def scope_url(self, **changes):
        packet = read_packet(self.work, "generic-pack")
        query = {
            "path": self.source_path, "topic": "/camera-x", "frame": 7,
            "file_activation_generation": 42,
            "expected_manifest_sha256": packet["manifest_sha256"],
        }
        query.update(changes)
        from urllib.parse import urlencode
        return ("/api/candidate_comparison_packs/generic-pack/"
                "any-frame-any-object/scope?" + urlencode(query))

    def test_generic_detail_is_sanitized_and_all_routes_are_get_only(self):
        client = server.app.test_client()
        listing = client.get("/api/candidate_comparison_packs")
        detail = client.get("/api/candidate_comparison_packs/generic-pack")
        self.assertEqual(listing.status_code, 200, listing.json)
        self.assertEqual(listing.json["schema_version"],
                         "labelagent.candidate-comparison-pack-index.v1")
        self.assertFalse(listing.json["human_truth_writes"])
        self.assertEqual(detail.status_code, 200, detail.json)
        serialized = json.dumps(detail.json)
        self.assertNotIn(self.source_path, serialized)
        self.assertNotIn(str(self.work), serialized)
        self.assertEqual(detail.json["rows"][0]["current_candidate_id"], 991)
        self.assertEqual(detail.json["rows"][0]["variants"][0]["alternative_id"],
                         "policy-alpha")
        for url in ("/api/candidate_comparison_packs",
                    "/api/candidate_comparison_packs/generic-pack",
                    "/api/candidate_comparison_packs/generic-pack/images/current.png"):
            self.assertEqual(client.post(url).status_code, 405)

    def test_image_and_scope_are_exactly_hash_bound(self):
        client = server.app.test_client()
        detail = client.get("/api/candidate_comparison_packs/generic-pack").json
        image_url = detail["rows"][0]["current"]["image_url"]
        image = client.get(image_url)
        self.assertEqual(image.status_code, 200)
        self.assertEqual(image.data, self.current_image)
        self.assertIn("no-store", image.headers["Cache-Control"])
        scope = client.get(self.scope_url())
        self.assertEqual(scope.status_code, 200, scope.json)
        self.assertTrue(scope.json["scope_verified"])
        self.assertFalse(scope.json["human_truth_writes"])
        self.assertNotIn(self.source_path, json.dumps(scope.json))
        stale_url = image_url.replace(
            "image_sha256=" + hashlib.sha256(self.current_image).hexdigest(),
            "image_sha256=" + "f" * 64)
        self.assertEqual(client.get(stale_url).status_code, 409)
        self.assertEqual(client.get(
            "/api/candidate_comparison_packs/generic-pack/images/current.png"
        ).status_code, 400)

    def test_scope_fails_closed_on_every_live_binding(self):
        client = server.app.test_client()
        cases = (
            ("path", {"path": self.source_path + ".other"}),
            ("topic", {"topic": "/other"}),
            ("frame", {"frame": 8}),
            ("generation", {"file_activation_generation": 41}),
            ("manifest", {"expected_manifest_sha256": "f" * 64}),
        )
        for name, changes in cases:
            with self.subTest(name=name):
                response = client.get(self.scope_url(**changes))
                self.assertEqual(response.status_code, 409, response.json)
        self.state["video"].payload = png_bytes((1, 2, 3))
        self.assertEqual(client.get(self.scope_url()).status_code, 409)
        self.state["video"].payload = self.frame_payload
        self.state["candidates"]["7"][0]["score"] = 0.4
        self.assertEqual(client.get(self.scope_url()).status_code, 409)

    def test_scope_independently_rejects_source_and_candidate_fingerprint_drift(self):
        client = server.app.test_client()
        changed = copy.deepcopy(self.value)
        changed["source"]["path"] = str((self.work / "other.mcap").resolve())
        self.manifest.write_text(json.dumps(changed), encoding="utf-8")
        packet = read_packet(self.work, "generic-pack")
        self.assertEqual(client.get(self.scope_url(
            expected_manifest_sha256=packet["manifest_sha256"])).status_code, 409)

        self.state["candidates"]["7"][0]["score"] = 0.4
        binding = server.candidate_review_binding(self.state["candidates"]["7"])
        changed = copy.deepcopy(self.value)
        # Keep the old per-candidate fingerprint but update the whole-artifact
        # hash so this test reaches the independent fingerprint gate.
        changed["rows"][0]["current_candidate_artifact_sha256"] = binding["candidate_artifact_sha256"]
        self.manifest.write_text(json.dumps(changed), encoding="utf-8")
        packet = read_packet(self.work, "generic-pack")
        self.assertEqual(client.get(self.scope_url(
            expected_manifest_sha256=packet["manifest_sha256"])).status_code, 409)

    def test_manifest_flags_paths_duplicates_and_tamper_fail_closed(self):
        for field, promoted in (("candidate_only", False), ("human_truth", True),
                                ("approved_for_training", True),
                                ("training_authorized", True),
                                ("automatic_approval", True)):
            changed = copy.deepcopy(self.value)
            changed[field] = promoted
            self.manifest.write_text(json.dumps(changed), encoding="utf-8")
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "隔离门禁"):
                read_packet(self.work, "generic-pack")
        changed = copy.deepcopy(self.value)
        changed["rows"][0]["variants"][0]["preview"]["image"] = "../escape.png"
        self.manifest.write_text(json.dumps(changed), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "图片绑定"):
            read_packet(self.work, "generic-pack")
        self.write_manifest()
        (self.packet / "alternative.png").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "哈希不一致"):
            read_packet(self.work, "generic-pack")

    def test_get_routes_do_not_mutate_candidate_or_truth_state(self):
        self.state["truth"] = {"7": [{"label": "human"}]}
        before = copy.deepcopy(self.state)
        client = server.app.test_client()
        self.assertEqual(client.get("/api/candidate_comparison_packs").status_code, 200)
        self.assertEqual(client.get("/api/candidate_comparison_packs/generic-pack").status_code, 200)
        self.assertEqual(client.get(self.scope_url()).status_code, 200)
        self.assertEqual(self.state["truth"], before["truth"])
        self.assertEqual(self.state["candidates"], before["candidates"])


if __name__ == "__main__":
    unittest.main()
