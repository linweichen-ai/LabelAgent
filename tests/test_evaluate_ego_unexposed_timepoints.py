import unittest

from tools.evaluate_ego_unexposed_timepoints import select_even_unexposed


class EgoUnexposedTimepointsTest(unittest.TestCase):
    def test_even_selection_is_deterministic_and_excludes_exposed(self):
        selected = select_even_unexposed(20, (0, 5, 10, 19), 6)
        self.assertEqual(selected, select_even_unexposed(20, (0, 5, 10, 19), 6))
        self.assertEqual(len(selected), 6)
        self.assertEqual(len(set(selected)), 6)
        self.assertTrue(set(selected).isdisjoint({0, 5, 10, 19}))

    def test_single_sample_uses_middle_available_frame(self):
        self.assertEqual(select_even_unexposed(5, (0, 4), 1), (2,))

    def test_insufficient_scope_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "not enough"):
            select_even_unexposed(3, (0, 1), 2)


if __name__ == "__main__":
    unittest.main()
