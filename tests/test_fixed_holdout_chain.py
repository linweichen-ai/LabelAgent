import json
from pathlib import Path
import tempfile
import unittest

from tools.run_fixed_holdout_chain import agent_output, canonical


class FixedHoldoutChainTests(unittest.TestCase):
    def test_no_op_preserves_candidate_bytes_and_hash_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "candidate.json"
            original = b'{\n  "parts": [{"points": [[1, 2], [3, 4], [5, 6]]}],\n  "score": 0.9\n}\n'
            path.write_bytes(original)
            parsed = json.loads(original)
            candidate, output = agent_output("no_op", path, parsed, None)
            self.assertEqual(candidate, parsed)
            self.assertEqual(output, original)

    def test_drop_is_explicit_empty_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "candidate.json"
            path.write_text("{}", encoding="utf-8")
            candidate, output = agent_output("drop", path, {}, None)
            self.assertEqual(candidate["parts"], [])
            self.assertEqual(output, canonical(candidate))


if __name__ == "__main__":
    unittest.main()
