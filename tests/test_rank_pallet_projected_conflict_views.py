import unittest

import numpy as np

from tools.rank_pallet_projected_conflict_views import (
    conflict_metrics, select_views,
)


def row(obj_id, x0, x1):
    mask = np.zeros((8, 16), dtype=bool)
    mask[2:6, x0:x1] = True
    return {"obj_id": obj_id, "mask": mask}


class ProjectedConflictRankingTests(unittest.TestCase):
    def test_transitive_component_metrics(self):
        metrics = conflict_metrics([
            row(1, 0, 5), row(2, 3, 8), row(3, 6, 11), row(4, 13, 16),
        ])
        self.assertEqual(metrics["maximum_component_size"], 3)
        self.assertEqual(metrics["maximum_component_edge_count"], 2)
        self.assertEqual(metrics["conflict_component_count"], 1)
        self.assertEqual(metrics["components"][0]["object_ids"], ["1", "2", "3"])

    def test_selection_prioritizes_unseen_sequences_and_is_deterministic(self):
        rows = []
        for sequence, size, edges, frame in (
                ("000", 9, 12, "a"), ("001", 3, 2, "b"),
                ("002", 7, 8, "c"), ("001", 2, 1, "d")):
            rows.append({
                "sequence": sequence, "frame_id": frame, "camera": 0,
                "maximum_component_size": size,
                "maximum_component_edge_count": edges,
                "conflict_edge_count": edges,
                "conflicted_object_count": size,
                "visible_pallet_count": size,
            })
        first = select_views(rows, 2, {"000", "002"})
        second = select_views(list(reversed(rows)), 2, {"000", "002"})
        self.assertEqual(first, second)
        self.assertEqual(first[0]["sequence"], "001")
        self.assertEqual(first[0]["frame_id"], "b")
        self.assertEqual(first[1]["sequence"], "000")


if __name__ == "__main__":
    unittest.main()
