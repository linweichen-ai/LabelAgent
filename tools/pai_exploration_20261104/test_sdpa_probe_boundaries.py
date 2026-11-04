import unittest
from test_qwen2b_sdpa import parse_answer, blank_passed, gpu_candidates


class ProbeBoundaries(unittest.TestCase):
    def test_busy_gpu_is_not_selected(self):
        self.assertEqual(gpu_candidates("0, 90, 60000\n1, 0, 20000\n"), 1)
    def test_no_idle_gpu_stops(self):
        with self.assertRaises(ValueError): gpu_candidates("0, 60, 70000\n1, 0, 1000\n")
    def test_reserved_gpu_is_not_used_as_fallback(self):
        with self.assertRaises(ValueError): gpu_candidates("0, 60, 70000\n1, 0, 70000\n", required_index=0)
    def test_blank_accept_does_not_pass(self):
        self.assertFalse(blank_passed(parse_answer('{"decision":"keep","label":"pallet"}')))
        self.assertTrue(blank_passed(parse_answer('{"decision":"uncertain","label":"unknown"}')))
    def test_malformed_and_inconsistent_answer_rejected(self):
        for text in ['```json\n{}\n```','{"decision":"reject","label":"person"}','{"decision":"keep","label":"pallet","extra":1}']:
            self.assertIsNone(parse_answer(text))


if __name__ == "__main__": unittest.main()
