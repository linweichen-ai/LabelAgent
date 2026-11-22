import hashlib
import os
from pathlib import Path
import subprocess
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "tools" / "start_local.ps1"


@unittest.skipUnless(os.name == "nt", "Windows named mutex acceptance")
class StartupMutexTests(unittest.TestCase):
    def test_second_start_fails_fast_and_release_restores_preflight(self):
        suffix = hashlib.sha256(str(ROOT.resolve()).lower().encode("utf-8")).hexdigest().upper()[:24]
        name = f"Local\\LabelAgentStart-{suffix}"
        holder_code = (
            f'$m=[Threading.Mutex]::new($false,"{name}");'
            '$ok=$m.WaitOne(0);if(-not $ok){exit 2};'
            '[Console]::Out.WriteLine("READY");[Console]::Out.Flush();'
            'Start-Sleep -Seconds 4;$m.ReleaseMutex();$m.Dispose()'
        )
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        holder = subprocess.Popen(
            ["powershell", "-NoProfile", "-Command", holder_code],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            encoding="mbcs", errors="replace",
            creationflags=flags,
        )
        try:
            self.assertEqual(holder.stdout.readline().strip(), "READY")
            started = time.monotonic()
            blocked = subprocess.run([
                "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                "-File", str(START), "-NoBrowser",
            ], encoding="mbcs", errors="replace", capture_output=True,
                timeout=10, creationflags=flags)
            self.assertNotEqual(blocked.returncode, 0)
            self.assertLess(time.monotonic() - started, 5)
            self.assertIn(
                "Another LabelAgent startup action is already running for this checkout",
                blocked.stdout + blocked.stderr,
            )
            self.assertEqual(holder.wait(timeout=8), 0)
            released = subprocess.run([
                "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                "-File", str(START), "-NoBrowser",
            ], encoding="mbcs", errors="replace", capture_output=True,
                timeout=15, creationflags=flags)
            self.assertNotEqual(released.returncode, 0)
            self.assertNotIn(
                "Another LabelAgent startup action is already running for this checkout",
                released.stdout + released.stderr,
            )
        finally:
            if holder.poll() is None:
                holder.terminate()
                holder.wait(timeout=5)
            holder.stdout.close()
            holder.stderr.close()


if __name__ == "__main__":
    unittest.main()
