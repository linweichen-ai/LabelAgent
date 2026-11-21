import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from tools.build_ego_component_union_review import (
    build_packet, evaluate_sample, file_sha256)


def _write_mask(path: Path, points: list[tuple[int, int]]) -> dict:
    mask = np.zeros((8, 10), dtype=np.uint8)
    for x, y in points:
        mask[y, x] = 255
    Image.fromarray(mask, mode="L").save(path)
    rows, columns = np.nonzero(mask)
    return {
        "area": int((mask > 0).sum()),
        "bbox": [int(columns.min()), int(rows.min()),
                 int(columns.max()) + 1, int(rows.max()) + 1],
    }


class EgoComponentUnionReviewTests(unittest.TestCase):
    def _fixture(self, root: Path, *, available=True, bad_hash=False):
        specs = {}
        for camera in ("camera1", "camera3"):
            path = root / f"{camera}-spec.json"
            path.write_text(f'{{"camera":"{camera}"}}\n', encoding="utf-8")
            specs[camera] = path
        image = root / "003-camera1-idx5-test.png"
        Image.new("RGB", (10, 8), (80, 90, 100)).save(image)
        sample = root / "input" / "003-camera1-idx5-test"
        sample.mkdir(parents=True)
        names = ("camera1-left-seat-console-v2", "camera1-right-hood-v2")
        points = ([(1, 1), (2, 1)], [(7, 5), (8, 5)])
        rows = []
        for index, (name, mask_points) in enumerate(zip(names, points)):
            mask_path = sample / f"{index:02d}-{name}-mask.png"
            geometry = _write_mask(mask_path, mask_points)
            rows.append({
                "name": name,
                "candidate_available": available,
                "prompt_constraints_passed": available,
                "candidate_mask": mask_path.name,
                "candidate_mask_sha256": (
                    "0" * 64 if bad_hash and index == 0 else file_sha256(mask_path)),
                "geometry": geometry,
            })
        report = {
            "candidate_only": True,
            "human_truth": False,
            "approved_for_training": False,
            "production_deployed": False,
            "source_state_modified": False,
            "point_sets_sha256": file_sha256(specs["camera1"]),
            "image": str(image),
            "image_sha256": file_sha256(image),
            "rows": rows,
        }
        (sample / "results.json").write_text(
            json.dumps(report), encoding="utf-8")
        return sample, specs

    def test_builds_union_only_after_all_hash_and_constraint_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample, specs = self._fixture(root)
            output = root / "output"
            output.mkdir()
            row = evaluate_sample(sample, output, {
                camera: file_sha256(path) for camera, path in specs.items()})
            self.assertTrue(row["union_available"])
            self.assertEqual(row["union_geometry"]["area_pixels"], 4)
            self.assertTrue((output / row["union_mask"]).is_file())

    def test_fails_closed_when_component_or_mask_hash_is_invalid(self):
        for available, bad_hash, reason in (
                (False, False, "required component failed closed"),
                (True, True, "hash-mismatched")):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                sample, specs = self._fixture(
                    root, available=available, bad_hash=bad_hash)
                output = root / "output"
                output.mkdir()
                row = evaluate_sample(sample, output, {
                    camera: file_sha256(path) for camera, path in specs.items()})
                self.assertFalse(row["union_available"])
                self.assertIn(reason, row["reasons"][0])

    def test_packet_keeps_candidates_out_of_truth_and_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, specs = self._fixture(root)
            packet = build_packet(root / "input", root / "packet",
                                  specs["camera1"], specs["camera3"])
            self.assertEqual(packet["union_available_count"], 1)
            self.assertFalse(packet["human_truth"])
            self.assertFalse(packet["approved_for_training"])
            self.assertTrue((root / "packet" / "review_queue.json").is_file())


if __name__ == "__main__":
    unittest.main()
