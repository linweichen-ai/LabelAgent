from argparse import Namespace
from pathlib import Path
import tempfile
import unittest

from tools.closed_loop_register_image import register_image, sha256_file


class ClosedLoopRegisterImageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.image = self.root / "frame.png"
        self.image.write_bytes(b"\x89PNG\r\n\x1a\nfixture")

    def args(self, expected=None):
        return Namespace(
            image=str(self.image), dataset_id="sfl-cpd15y",
            source_batch="20260920_cs-heli_Dev_000001/000",
            frame_id="1789897506782450197", camera="camera_0",
            captured_at="2026-09-20T09:45:06.782450197Z",
            source_uri="oss://example/batch/000/frame.png",
            timestamp_source="frame_id_unix_ns", expected_sha256=expected,
            ledger_root=str(self.root / "ledger"), output=str(self.root / "bundle.json"),
        )

    def test_registers_only_received_and_is_idempotent(self):
        digest = sha256_file(self.image)
        first = register_image(self.args(digest))
        self.assertTrue(first["registration"]["created"])
        self.assertEqual([event["state"] for event in first["events"]], ["received"])
        self.assertFalse(first["receipt"]["minimum_closed_loop_complete"])
        self.assertFalse(first["evidence_boundary"]["model_events_recorded"])
        replay_args = self.args(digest)
        replay_args.output = str(self.root / "replay-bundle.json")
        second = register_image(replay_args)
        self.assertTrue(second["registration"]["idempotent_replay"])
        self.assertEqual(len(second["events"]), 1)

    def test_rejects_hash_mismatch(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            register_image(self.args("0" * 64))


if __name__ == "__main__":
    unittest.main()
