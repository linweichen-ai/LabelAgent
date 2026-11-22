import json
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "write_process_record_atomic.ps1"


def payload(startup_id: str) -> str:
    return json.dumps({
        "schema_version": "labelagent.local-process-record.v2",
        "startup_id": startup_id,
        "processes": [{
            "pid": 101,
            "role": "workbench",
            "started_at_utc": "2026-11-22T00:00:00Z",
            "executable_path": "C:\\example\\python.exe",
        }],
        "pids": [101],
        "started_at": "2026-11-22T00:00:01Z",
        "sam_url": "http://127.0.0.1:8766",
        "root": "C:\\example",
    }, separators=(",", ":"))


class AtomicProcessRecordTests(unittest.TestCase):
    def run_writer(self, record: Path, content: str):
        return subprocess.run([
            "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-File", str(SCRIPT), "-Record", str(record), "-Json", content,
        ], text=True, capture_output=True, timeout=20)

    def test_creates_and_replaces_record_without_pending_files(self):
        with tempfile.TemporaryDirectory() as directory:
            repro = Path(directory) / ".repro"
            repro.mkdir()
            record = repro / "processes.json"
            first = payload("first")
            second = payload("second")
            self.assertEqual(self.run_writer(record, first).returncode, 0)
            self.assertEqual(record.read_text(encoding="utf-8"), first)
            self.assertEqual(self.run_writer(record, second).returncode, 0)
            self.assertEqual(record.read_text(encoding="utf-8"), second)
            self.assertEqual(list(repro.glob("processes.pending-*.json")), [])
            self.assertEqual(list(repro.glob("processes.backup-*.json")), [])

    def test_invalid_payload_preserves_previous_record(self):
        with tempfile.TemporaryDirectory() as directory:
            repro = Path(directory) / ".repro"
            repro.mkdir()
            record = repro / "processes.json"
            original = payload("original")
            record.write_text(original, encoding="utf-8")
            result = self.run_writer(record, "{")
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(record.read_text(encoding="utf-8"), original)
            self.assertEqual(list(repro.glob("processes.pending-*.json")), [])

    def test_refuses_a_record_outside_dot_repro(self):
        with tempfile.TemporaryDirectory() as directory:
            record = Path(directory) / "processes.json"
            result = self.run_writer(record, payload("outside"))
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(record.exists())


if __name__ == "__main__":
    unittest.main()
