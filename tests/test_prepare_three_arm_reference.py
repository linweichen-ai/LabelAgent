import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tools.build_mixed_hard_adjudication_workspace import (
    LABELS, SUBMISSION_SCHEMA, sha256, validate_packet,
)
from tools.prepare_three_arm_reference import build_reference


class PrepareThreeArmReferenceTest(unittest.TestCase):
    def _packet(self, root):
        root = Path(root)
        (root / "raw").mkdir(parents=True)
        (root / "review").mkdir()
        rows, files = [], []
        for index in range(24):
            for folder in ("raw", "review"):
                path = root / folder / f"{index}.jpg"
                Image.new("RGB", (8, 8), (index, index, 0)).save(path)
                files.append({"path": path.relative_to(root).as_posix(),
                              "sha256": sha256(path), "bytes": path.stat().st_size})
            rows.append({"sample_id": f"sample-{index}", "cohort": "fixture",
                         "source_image_sha256": f"{index + 1:064x}",
                         "raw_image": f"raw/{index}.jpg",
                         "review_image": f"review/{index}.jpg",
                         "candidate_only": True, "human_truth": False,
                         "approved_for_training": False})
        manifest = {"schema_version": "labelagent.mixed-hard-review-pack.v1",
                    "summary": {"test24_intersection_count": 0}, "rows": rows,
                    "candidate_only": True, "human_truth": False,
                    "approved_for_training": False, "training_authorized": False,
                    "test24_used": False, "production_deployed": False}
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        (root / "receipt.json").write_text(json.dumps({
            "manifest_sha256": sha256(manifest_path), "files": files}), encoding="utf-8")
        return root

    def _submission(self, manifest, digest):
        submission = {
            "schema_version": SUBMISSION_SCHEMA,
            "review_manifest_sha256": digest,
            "reviewer_declared_id": "reviewer-1",
            "rows": [{"sample_id": row["sample_id"],
                      "source_image_sha256": row["source_image_sha256"],
                      "candidate_decision": "uncertain", "slope_presence": "absent",
                      "class_conflict": "none", "pallet_quality": "not_applicable"}
                     for row in manifest["rows"]],
            "candidate_only": True, "human_truth": False,
            "approved_for_training": False, "training_authorized": False,
            "automatic_approval": False,
        }
        for row in submission["rows"]:
            row["class_presence"] = {label: "absent" for label in LABELS}
            row["class_presence"]["floor"] = "present"
            row["pallet_instance_count"] = 0
        return submission

    def test_prepares_bound_reference_without_runs_or_training_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            packet = self._packet(Path(directory) / "packet")
            manifest, digest = validate_packet(packet)
            payload = build_reference(packet, self._submission(manifest, digest))
            self.assertEqual(len(payload["samples"]), 24)
            self.assertEqual(payload["runs"], [])
            self.assertTrue(payload["reviewer_confirmed"])
            self.assertFalse(payload["pixel_truth"])
            self.assertFalse(payload["approved_for_training"])
            self.assertEqual(payload["samples"][0]["source_image_sha256"],
                             manifest["rows"][0]["source_image_sha256"])

    def test_rejects_uncertain_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            packet = self._packet(Path(directory) / "packet")
            manifest, digest = validate_packet(packet)
            submission = self._submission(manifest, digest)
            submission["rows"][0]["class_presence"]["slope"] = "uncertain"
            with self.assertRaisesRegex(ValueError, "not benchmark-ready"):
                build_reference(packet, submission)


if __name__ == "__main__":
    unittest.main()
