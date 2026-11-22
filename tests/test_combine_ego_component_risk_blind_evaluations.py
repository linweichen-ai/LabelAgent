import json
from pathlib import Path
import tempfile
import unittest

from tools.combine_ego_component_risk_blind_evaluations import combine


def packet(prefix, reject_bottom=False):
    rows = []
    index = 0
    for camera in ("camera1", "camera3"):
        for band in ("top", "middle", "bottom"):
            index += 1
            decision = "likely_fill_ok"
            if band in {"top", "middle"} or reject_bottom:
                decision = "likely_reject"
            rows.append({
                "audit_id": f"A{index:03d}",
                "component_id": f"{prefix}-{camera}-{band}",
                "camera": camera,
                "hidden_band": band,
                "decision": decision,
            })
    return {
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "production_deployed": False,
        "default_enabled": False,
        "assistant_review_only": True,
        "reviewer_blinded_to_risk_band": True,
        "selected_count": len(rows),
        "revealed_rows": rows,
    }


class CombinedBlindEvaluationTests(unittest.TestCase):
    def test_combines_disjoint_evaluations_and_preserves_safety_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / "a.json", root / "b.json"]
            for path, payload in zip(paths, (packet("a"), packet("b"))):
                path.write_text(json.dumps(payload), encoding="utf-8")
            output = root / "combined.json"
            result = combine(paths, output)
            self.assertEqual(result["selected_count"], 12)
            self.assertEqual(result["likely_reject_total"], 8)
            self.assertEqual(
                result["top_plus_middle_likely_reject_capture_rate"], 1.0)
            self.assertEqual(result["bottom_likely_reject_count"], 0)
            self.assertFalse(result["human_truth"])
            self.assertTrue(output.exists())

    def test_rejects_overlapping_component_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / "a.json", root / "b.json"]
            payload = packet("same")
            for path in paths:
                path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "multiple evaluations"):
                combine(paths, root / "combined.json")

    def test_rejects_unsafe_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / "a.json", root / "b.json"]
            first, second = packet("a"), packet("b")
            second["human_truth"] = True
            for path, payload in zip(paths, (first, second)):
                path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "human_truth"):
                combine(paths, root / "combined.json")

    def test_requires_two_evaluations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "a.json"
            path.write_text(json.dumps(packet("a")), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "at least two"):
                combine([path], root / "combined.json")


if __name__ == "__main__":
    unittest.main()
