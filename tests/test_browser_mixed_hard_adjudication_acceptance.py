import json
import tempfile
import unittest
from pathlib import Path

from tests.test_build_mixed_hard_adjudication_workspace import (
    MixedHardAdjudicationWorkspaceTest,
)
from tools.browser_mixed_hard_adjudication_acceptance import (
    validate_workspace,
)
from tools.build_mixed_hard_adjudication_workspace import build_workspace


class MixedHardBrowserAcceptanceTest(unittest.TestCase):
    def workspace(self, root: Path):
        packet = MixedHardAdjudicationWorkspaceTest().packet(root / "packet")
        workspace = root / "workspace"
        build_workspace(packet, workspace)
        receipt_path = workspace / "receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["assistant_hint_blind_gate"] = True
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        return workspace

    def test_validates_hash_bound_candidate_only_blind_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = self.workspace(Path(directory))
            result = validate_workspace(workspace)
            self.assertEqual(len(result["files"]), 50)
            self.assertIn("reviewer.html", result["files"])
            self.assertFalse(result["receipt"]["human_truth"])

    def test_rejects_missing_blind_gate_hash_drift_and_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = self.workspace(root)
            receipt_path = workspace / "receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["assistant_hint_blind_gate"] = False
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "assistant_hint_blind_gate"):
                validate_workspace(workspace)

            workspace = self.workspace(root / "hash")
            (workspace / "reviewer.html").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                validate_workspace(workspace)

            workspace = self.workspace(root / "escape")
            receipt_path = workspace / "receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["files"]["../outside"] = "0" * 64
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsafe or invalid"):
                validate_workspace(workspace)


if __name__ == "__main__":
    unittest.main()
