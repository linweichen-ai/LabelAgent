from copy import deepcopy
import unittest

from tools.audit_training_data_gate import audit_training_gate


def fixture():
    images = []
    image_id = 1
    for split, sequences, per_sequence in (
        ("train", range(9), 8),
        ("test", range(10, 13), 8),
    ):
        for sequence in sequences:
            for index in range(per_sequence):
                name = f"{sequence:03d}_{index:02d}_camera_{index % 4}.png"
                images.append({
                    "image_id": image_id,
                    "split": split,
                    "sequence": f"{sequence:03d}",
                    "file_name": name,
                    "source_sha256": f"{image_id:064x}",
                })
                image_id += 1
    manifest = {
        "image_count": 96,
        "train_images": 72,
        "test_images": 24,
        "preannotations_are_ground_truth": False,
        "images": images,
    }
    validation = {"verified_image_hashes": 96, "human_annotations_present": False, "training_authorized": False}
    tasks = {}
    for split, count, task_id in (("train", 72, 445), ("test", 24, 446)):
        tasks[split] = {
            "task_id": task_id,
            "creation_completed": True,
            "readback": {
                "subset": split,
                "size": count,
                "frame_count": count,
                "labels": [{"name": "pallet", "type": "mask"}, {"name": "person", "type": "mask"}],
                "preannotations_are_human_truth": False,
                "validation_errors": [],
                "job_ids": [task_id - 10],
                "shape_count": count * 3,
            },
        }
    receipt = {
        "status": "complete",
        "train_test_isolation": True,
        "machine_preannotations_only": True,
        "tasks": tasks,
    }
    checklist = [
        {"split": image["split"], "file_name": image["file_name"], "review_status": "pending", "reviewer": ""}
        for image in images
    ]
    return manifest, validation, receipt, checklist


def human_coco(manifest, split):
    selected = [image for image in manifest["images"] if image["split"] == split]
    images = [{"id": index + 1, "file_name": image["file_name"]} for index, image in enumerate(selected)]
    return {
        "info": {
            "annotation_status": "human_verified",
            "reviewer": "reviewer",
            "reviewed_image_ids": [image["id"] for image in images],
        },
        "images": images,
        "categories": [{"id": 1, "name": "pallet"}, {"id": 2, "name": "person"}],
        "annotations": [
            {"id": index + 1, "image_id": image["id"], "category_id": 1, "segmentation": [[0, 0, 1, 0, 1, 1]]}
            for index, image in enumerate(images)
        ],
    }


class TrainingDataGateTests(unittest.TestCase):
    def test_machine_preannotations_never_authorize_training(self):
        manifest, validation, receipt, checklist = fixture()
        result = audit_training_gate(manifest, validation, receipt, checklist, {"train": None, "test": None})
        self.assertFalse(result["training_authorized"])
        self.assertEqual(result["status"], "blocked_waiting_for_human_review")
        self.assertEqual(result["checklist"]["pending_or_invalid"], 96)
        self.assertEqual(result["split_isolation"]["file_name_overlap"], [])

    def test_complete_human_review_opens_only_isolated_training_gate(self):
        manifest, validation, receipt, checklist = fixture()
        for row in checklist:
            row["review_status"] = "approved"
            row["reviewer"] = "reviewer"
        result = audit_training_gate(
            manifest,
            validation,
            receipt,
            checklist,
            {"train": human_coco(manifest, "train"), "test": human_coco(manifest, "test")},
        )
        self.assertTrue(result["training_authorized"])
        self.assertEqual(result["status"], "ready_for_isolated_training")
        self.assertEqual(result["test_split_policy"], "evaluation_only_never_train")

    def test_partial_export_or_missing_reviewer_stays_blocked(self):
        manifest, validation, receipt, checklist = fixture()
        for row in checklist:
            row["review_status"] = "approved"
            row["reviewer"] = "reviewer"
        train = human_coco(manifest, "train")
        train["info"]["reviewer"] = ""
        result = audit_training_gate(manifest, validation, receipt, checklist, {"train": train, "test": None})
        self.assertFalse(result["training_authorized"])
        self.assertTrue(any("reviewer is missing" in blocker for blocker in result["blockers"]))
        self.assertFalse(result["human_exports"]["present"]["test"])

    def test_split_leakage_fails_closed_even_with_human_exports(self):
        manifest, validation, receipt, checklist = fixture()
        leaked = deepcopy(manifest["images"][0])
        target = next(image for image in manifest["images"] if image["split"] == "test")
        target["file_name"] = leaked["file_name"]
        target["source_sha256"] = leaked["source_sha256"]
        checklist = [
            {"split": image["split"], "file_name": image["file_name"], "review_status": "approved", "reviewer": "reviewer"}
            for image in manifest["images"]
        ]
        result = audit_training_gate(
            manifest,
            validation,
            receipt,
            checklist,
            {"train": human_coco(manifest, "train"), "test": human_coco(manifest, "test")},
        )
        self.assertFalse(result["training_authorized"])
        self.assertTrue(result["split_isolation"]["file_name_overlap"])
        self.assertTrue(result["split_isolation"]["source_sha256_overlap"])


if __name__ == "__main__":
    unittest.main()
