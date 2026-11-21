import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from tools.build_ego_component_union_review import (
    build_packet, evaluate_sample, file_sha256, load_component_guard)


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
    @staticmethod
    def _guard(minimum=0.0, maximum=1.0):
        return {
            name: {
                "min_area_ratio": minimum,
                "max_area_ratio": maximum,
            }
            for names in (
                ("camera1-left-seat-console-v2",
                 "camera1-right-hood-v2"),
                ("camera3-crossbar-core-v2",
                 "camera3-lower-assembly-v2"),
            )
            for name in names
        }

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

    def test_packet_records_enabled_safety_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, specs = self._fixture(root)
            guard_path = root / "guard.json"
            guard_path.write_text(
                json.dumps({"components": self._guard()}), encoding="utf-8")
            packet = build_packet(
                root / "input", root / "packet",
                specs["camera1"], specs["camera3"],
                allowed_source_root=root,
                component_guard_path=guard_path)
            self.assertEqual(
                packet["safety_gates"]["allowed_source_root"],
                str(root.resolve()))
            self.assertTrue(
                packet["safety_gates"]["component_area_guard_enabled"])
            self.assertEqual(
                packet["safety_gates"]["component_area_guard_sha256"],
                file_sha256(guard_path))

    def test_allowed_source_root_accepts_contained_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample, specs = self._fixture(root)
            output = root / "output"
            output.mkdir()
            row = evaluate_sample(
                sample, output,
                {camera: file_sha256(path) for camera, path in specs.items()},
                allowed_source_root=root)
            self.assertTrue(row["union_available"])

    def test_allowed_source_root_rejects_outside_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample, specs = self._fixture(root)
            allowed = root / "allowed"
            allowed.mkdir()
            output = root / "output"
            output.mkdir()
            row = evaluate_sample(
                sample, output,
                {camera: file_sha256(path) for camera, path in specs.items()},
                allowed_source_root=allowed)
            self.assertFalse(row["union_available"])
            self.assertIn("outside allowed source root", row["reasons"][0])

    def test_allowed_source_root_rejects_parent_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample, specs = self._fixture(root)
            allowed = root / "allowed"
            allowed.mkdir()
            report_path = sample / "results.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["image"] = str(allowed / ".." / Path(report["image"]).name)
            report_path.write_text(json.dumps(report), encoding="utf-8")
            output = root / "output"
            output.mkdir()
            row = evaluate_sample(
                sample, output,
                {camera: file_sha256(path) for camera, path in specs.items()},
                allowed_source_root=allowed)
            self.assertFalse(row["union_available"])
            self.assertIn("outside allowed source root", row["reasons"][0])

    def test_allowed_source_root_rejects_symlink_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample, specs = self._fixture(root)
            allowed = root / "allowed"
            allowed.mkdir()
            link = allowed / "linked-image.png"
            try:
                link.symlink_to(root / "003-camera1-idx5-test.png")
            except OSError as error:
                self.skipTest(f"symlink creation unavailable: {error}")
            report_path = sample / "results.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["image"] = str(link)
            report_path.write_text(json.dumps(report), encoding="utf-8")
            output = root / "output"
            output.mkdir()
            row = evaluate_sample(
                sample, output,
                {camera: file_sha256(path) for camera, path in specs.items()},
                allowed_source_root=allowed)
            self.assertFalse(row["union_available"])
            self.assertIn("outside allowed source root", row["reasons"][0])

    def test_component_area_guard_accepts_inclusive_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample, specs = self._fixture(root)
            output = root / "output"
            output.mkdir()
            row = evaluate_sample(
                sample, output,
                {camera: file_sha256(path) for camera, path in specs.items()},
                component_guard=self._guard(0.025, 0.025))
            self.assertTrue(row["union_available"])
            self.assertEqual(
                row["components"][0]["area_guard"]["max_area_ratio"], 0.025)

    def test_component_area_guard_fails_closed_on_large_component(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample, specs = self._fixture(root)
            output = root / "output"
            output.mkdir()
            row = evaluate_sample(
                sample, output,
                {camera: file_sha256(path) for camera, path in specs.items()},
                component_guard=self._guard(0.0, 0.02))
            self.assertFalse(row["union_available"])
            self.assertIn("area ratio outside guard", row["reasons"][0])

    def test_component_area_guard_fails_closed_when_rule_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample, specs = self._fixture(root)
            guard = self._guard()
            del guard["camera1-right-hood-v2"]
            output = root / "output"
            output.mkdir()
            row = evaluate_sample(
                sample, output,
                {camera: file_sha256(path) for camera, path in specs.items()},
                component_guard=guard)
            self.assertFalse(row["union_available"])
            self.assertIn("area guard missing", row["reasons"][0])

    def test_malformed_component_guard_is_rejected_before_packet_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            guard_path = root / "guard.json"
            guard_path.write_text(
                json.dumps({"components": {"bad": {
                    "min_area_ratio": 0.5, "max_area_ratio": 0.1}}}),
                encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid component area guard"):
                load_component_guard(guard_path)


if __name__ == "__main__":
    unittest.main()
