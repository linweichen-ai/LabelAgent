import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.audit_vlm_teacher_candidates import audit_collection, json_sha256


STATE = "a" * 64
MODEL = "b" * 64


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(root: Path):
    images = []
    for index in range(72):
        sequence = index % 9
        name = f"{sequence:03d}_{index}.png"
        images.append({"split": "train", "file_name": name,
                       "source_sha256": f"{index + 1:064x}"})
    for index in range(24):
        images.append({"split": "test", "file_name": f"test_{index}.png",
                       "source_sha256": f"{index + 1000:064x}"})
    manifest = {"images": images}
    contract_body = {"schema_version": "contract", "labels": ["pallet"]}
    contract_sha = json_sha256(contract_body)
    write_json(root / "contract.json", contract_body | {"contract_sha256": contract_sha})
    for index, image in enumerate(images[:72], 1):
        target = {"assessments": {"pallet": {"evidence": "present"}}, "unknown_candidates": []}
        row = {
            "image": {"file_name": image["file_name"], "source_sha256": image["source_sha256"]},
            "split": "train", "candidate_kind": "machine_teacher_output",
            "human_truth": False, "training_authorized": False,
            "requires_acceptance_gate": True, "distillation_candidate": True,
            "bindings": {"state_sha256": STATE, "contract_sha256": contract_sha},
            "teacher": {"model_config_sha256": MODEL, "raw_parsed": {},
                        "raw_parsed_sha256": json_sha256({}),
                        "cache_hit": False, "coalesced": False},
            "target": target, "target_sha256": json_sha256(target),
            "review": {"open_set_scan_complete": True, "consistency_flags": [], "registry_conflicts": []},
        }
        write_json(root / "rows" / f"{index:03d}.json", row)
    write_json(root / "summary.json", {
        "test24_included": False, "human_truth": False, "training_authorized": False,
        "images": 72, "distillation_candidates": 72, "review_required": 0,
        "cache_hits": 0, "coalesced_requests": 0,
        "contract_sha256": contract_sha, "state_sha256": STATE,
        "teacher_model_config_sha256": MODEL,
    })
    return manifest


class TeacherCollectionAuditTests(unittest.TestCase):
    def test_exact_train72_collection_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = audit_collection(root, fixture(root), STATE, MODEL)
            self.assertEqual(result["status"], "passed")
            self.assertTrue(result["train72_exact_coverage"])
            self.assertFalse(result["test24_included"])

    def test_test_leakage_and_truth_promotion_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = fixture(root)
            path = root / "rows" / "001.json"
            row = json.loads(path.read_text(encoding="utf-8"))
            test = next(image for image in manifest["images"] if image["split"] == "test")
            row["image"] = {"file_name": test["file_name"], "source_sha256": test["source_sha256"]}
            row["human_truth"] = True
            write_json(path, row)
            result = audit_collection(root, manifest, STATE, MODEL)
            self.assertEqual(result["status"], "failed")
            self.assertTrue(any("test24 leakage" in error for error in result["errors"]))
            self.assertTrue(any("mislabeled as truth" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
