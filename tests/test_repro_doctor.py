import importlib.util
from pathlib import Path
import socket
import tempfile
import unittest
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "repro_doctor.py"
SPEC = importlib.util.spec_from_file_location("repro_doctor", MODULE_PATH)
doctor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(doctor)


class ReproDoctorTests(unittest.TestCase):
    def test_item_preserves_required_boundary(self):
        self.assertEqual(
            doctor.item("weight", False, "missing", required=False),
            {"name": "weight", "ok": False, "required": False, "detail": "missing"},
        )

    def test_url_json_reads_json(self):
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = b'{"status":"ok"}'
        with mock.patch.object(doctor.urllib.request, "urlopen", return_value=response):
            ok, detail = doctor.url_json("http://127.0.0.1/health")
        self.assertTrue(ok)
        self.assertIn('"status": "ok"', detail)

    def test_port_available_reports_occupied_port(self):
        with mock.patch.object(doctor.socket, "socket") as factory:
            factory.return_value.__enter__.return_value.bind.side_effect = OSError("busy")
            self.assertFalse(doctor.port_available(8765))

    def test_port_available_rejects_a_real_windows_listener(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            self.assertFalse(doctor.port_available(port))

    def test_report_uses_requested_workbench_port(self):
        args = mock.MagicMock(
            require_cuda=False, require_weight=False,
            work_dir=tempfile.gettempdir(), require_services=False,
            web_port=18765, sam_port=18766,
            require_web_port=True, require_sam_port=False,
        )
        with mock.patch.object(doctor, "port_available", return_value=True), \
                mock.patch.object(doctor, "import_check",
                                  return_value=doctor.item("dependency", True, "ok")):
            report = doctor.build_report(args)
        names = {item["name"] for item in report["checks"]}
        self.assertIn("port:18765", names)
        self.assertNotIn("port:8765", names)

    def test_work_directory_check_is_reversible(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / ".labelagent-write-check"
            marker.write_text("ok", encoding="utf-8")
            marker.unlink()
            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
