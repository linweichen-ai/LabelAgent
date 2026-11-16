from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.owned_process_guard import lease_stop_reason, process_identity, same_owned_process, signal_owned


class OwnedWorkerChecks(unittest.TestCase):
    def identity(self):
        return {"pid": 7, "start_ticks": 123, "argv": ["python", "/owned/server.py"], "uid": 1000}

    def test_pid_reuse_or_different_command_or_owner_never_matches(self):
        original = self.identity()
        for key, value in (("start_ticks", 124), ("argv", ["python", "/foreign/server.py"]), ("uid", 1001)):
            changed = deepcopy(original); changed[key] = value
            self.assertFalse(same_owned_process(changed, original))
        self.assertFalse(same_owned_process(None, original))

    def test_command_name_spaces_do_not_move_birth_time_field(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "7"; folder.mkdir()
            tail = ["S"] + ["0"] * 18 + ["123"]
            (folder / "stat").write_text("7 (python with spaces) " + " ".join(tail))
            (folder / "cmdline").write_bytes(b"python\0/owned/server.py\0")
            found = process_identity(7, directory)
            self.assertEqual(found["start_ticks"], 123)
            self.assertEqual(found["argv"], ["python", "/owned/server.py"])

    def test_signal_requires_fresh_matching_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "7"; folder.mkdir()
            (folder / "stat").write_text("7 (python) " + " ".join(["S"] + ["0"] * 18 + ["123"]))
            (folder / "cmdline").write_bytes(b"python\0/owned/server.py\0")
            expected = process_identity(7, directory); calls = []
            self.assertTrue(signal_owned(7, expected, 15, lambda *a: calls.append(a), directory))
            (folder / "stat").write_text("7 (python) " + " ".join(["S"] + ["0"] * 18 + ["124"]))
            self.assertFalse(signal_owned(7, expected, 15, lambda *a: calls.append(a), directory))
            self.assertEqual(calls, [(7, 15)])

    def test_deadline_is_inclusive(self):
        self.assertIsNone(lease_stop_reason(99, 100, 20000))
        self.assertEqual(lease_stop_reason(100, 100, 20000), "deadline")

    def test_headroom_and_foreign_job_stop_only_the_owned_lease(self):
        self.assertIsNone(lease_stop_reason(1, 100, 4096))
        self.assertEqual(lease_stop_reason(1, 100, 4095), "headroom_below_reserve")
        self.assertEqual(lease_stop_reason(1, 100, 20000, foreign_compute=True), "foreign_compute_started")

    def test_missing_process_cannot_be_signalled(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(process_identity(7, directory))
            self.assertFalse(signal_owned(7, self.identity(), 15, lambda *a: self.fail("Unexpected signal"), directory))

    def test_real_signal_path_holds_process_handle_before_identity_check(self):
        events = []
        with patch('tools.owned_process_guard.os.pidfd_open', create=True, side_effect=lambda pid: events.append('open') or 9), \
             patch('tools.owned_process_guard.process_identity', side_effect=lambda *a: events.append('identity') or self.identity()), \
             patch('tools.owned_process_guard.signal.pidfd_send_signal', create=True, side_effect=lambda *a: events.append('signal')), \
             patch('tools.owned_process_guard.os.close', side_effect=lambda fd: events.append('close')):
            self.assertTrue(signal_owned(7, self.identity(), 15))
        self.assertEqual(events, ['open', 'identity', 'signal', 'close'])


if __name__ == "__main__":
    unittest.main()
