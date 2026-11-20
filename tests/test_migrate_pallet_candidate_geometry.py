import importlib.util
from pathlib import Path
import unittest


PATH = (Path(__file__).resolve().parents[1] / "tools"
        / "migrate_pallet_candidate_geometry.py")
SPEC = importlib.util.spec_from_file_location(
    "migrate_pallet_candidate_geometry", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PalletCandidateGeometryMigrationTests(unittest.TestCase):
    def test_missing_agent_run_uses_server_null_binding(self):
        state = {
            "candidates": {"0": [{"id": 1, "label": "pallet"}]},
            "agent_runs": {},
        }
        binding = MODULE._frame_binding(0, state)
        self.assertIsNone(binding["agent_run_sha256"])
        self.assertEqual(binding["candidate_count"], 1)
        self.assertEqual(binding["pallet_candidate_count"], 1)

    def test_stored_empty_agent_run_keeps_hash_binding(self):
        state = {
            "candidates": {"0": [{"id": 1, "label": "pallet"}]},
            "agent_runs": {"0": {}},
        }
        binding = MODULE._frame_binding(0, state)
        self.assertEqual(
            binding["agent_run_sha256"], MODULE._canonical_sha({}))

    def test_frame_run_does_not_reinterpret_other_frame_mapping(self):
        run, present = MODULE._frame_run({"1": {"status": "complete"}}, 0)
        self.assertEqual(run, {})
        self.assertFalse(present)

    def test_identity_comparison_ignores_only_geometry_fields(self):
        before = [{
            "id": 4, "label": "pallet", "score": 0.9,
            "points": [[1, 2], [3, 4]],
            "holes": [[[2, 3], [3, 3]]],
            "parts": [{"points": [[1, 2]], "holes": []}],
        }]
        after = [{
            "id": 4, "label": "pallet", "score": 0.9,
            "points": [[1, 2], [5, 6]],
            "holes": [],
            "parts": [{"points": [[1, 2], [5, 6]], "holes": []}],
        }]
        self.assertEqual(
            MODULE._without_geometry(before), MODULE._without_geometry(after))
        after[0]["label"] = "floor"
        self.assertNotEqual(
            MODULE._without_geometry(before), MODULE._without_geometry(after))


if __name__ == "__main__":
    unittest.main()
