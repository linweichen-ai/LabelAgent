import unittest

from tools.compare_visible_discovery_routes import compare_routes


def row(digest, evidence, *, elapsed=10, visible=None, unknown=None,
        flags=None, open_set=True):
    return {
        "image": f"dataset/010/bevannotation_data/cameras_640x512/camera_0/{digest}.png",
        "image_sha256": digest,
        "elapsed_ms": elapsed,
        "failure": None,
        "assessments": [{"label": label, "evidence": value, "confidence": 0.9}
                        for label, value in evidence.items()],
        "visible_labels": visible or [],
        "unknown_candidates": unknown or [],
        "consistency_flags": flags or [],
        "open_set_scan_complete": open_set,
    }


class VisibleDiscoveryRouteComparisonTests(unittest.TestCase):
    def test_exact_faster_route_passes_bounded_gate(self):
        primary = [row("a", {"slope": "present"}, elapsed=20,
                       visible=["slope"])]
        fast = [row("a", {"slope": "present"}, elapsed=10,
                    visible=["slope"])]
        result = compare_routes(primary, fast)
        self.assertTrue(result["authoritative_replacement_accepted"])
        self.assertEqual(result["decision"],
                         "accepted_for_bounded_authoritative_route")
        self.assertEqual(result["median_speedup_pct"], 50.0)
        self.assertEqual(result["mismatches"], [])

    def test_present_miss_and_extra_fail_gate_with_bounded_names(self):
        primary = [row("a", {"other_robot": "present", "车辆": "absent"},
                       elapsed=20, visible=["other_robot"])]
        fast = [row("a", {"other_robot": "absent", "车辆": "present"},
                    elapsed=10, visible=["车辆"])]
        result = compare_routes(primary, fast)
        self.assertFalse(result["authoritative_replacement_accepted"])
        self.assertEqual(result["label_stats"]["other_robot"]
                         ["primary_present_fast_not"], 1)
        self.assertEqual(result["label_stats"]["车辆"]
                         ["primary_not_fast_present"], 1)
        self.assertEqual(result["mismatches"][0]["image"],
                         "010/camera_0/a.png")
        self.assertNotIn("dataset", result["mismatches"][0]["image"])

    def test_unknown_or_open_set_drift_fails_gate(self):
        primary = [row("a", {"floor": "present"}, elapsed=20,
                       visible=["floor"], unknown=[{"candidate_name": "rack"}])]
        fast = [row("a", {"floor": "present"}, elapsed=10,
                    visible=["floor"], unknown=[], open_set=False)]
        result = compare_routes(primary, fast)
        self.assertFalse(result["authoritative_replacement_accepted"])
        self.assertEqual(result["unknown_exact_count"], 0)
        self.assertFalse(result["mismatches"][0]["open_set_complete_exact"])

    def test_pairing_mismatch_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "image sets do not match"):
            compare_routes(
                [row("a", {"floor": "present"})],
                [row("b", {"floor": "present"})])


if __name__ == "__main__":
    unittest.main()
