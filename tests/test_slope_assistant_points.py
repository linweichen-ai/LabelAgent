import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.evaluate_slope_assistant_points import load_point_candidates


class SlopeAssistantPointTests(unittest.TestCase):
    def test_boundary_stays_candidate_only(self):
        rows = [{"sample_id": f"sample-{index}",
                 "assistant_visible_slope_candidate": index < 4}
                for index in range(8)]
        payload = {
            "schema_version": "labelagent.slope-assistant-point-candidates.v1",
            "status": "assistant_candidate_only", "split": "dev8",
            "source_sequence": "008", "human_truth": False,
            "human_confirmed": False, "training_authorized": False,
            "color_rule_used": False,
            "items": [{"sample_id": f"sample-{index}",
                       "points": [{"x": 10, "y": 10, "label": 1}]}
                      for index in range(4)],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "points.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(len(load_point_candidates(path, rows)), 4)
            payload["human_truth"] = True
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_point_candidates(path, rows)


if __name__ == "__main__":
    unittest.main()
