#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
RECORD="$ROOT/.repro/linux-processes.json"
[[ "$(uname -s)" == "Linux" ]] || { echo "Linux only" >&2; exit 2; }
[[ -x "$PY" ]] || { echo "Missing .venv" >&2; exit 2; }
[[ -f "$RECORD" ]] || { echo "LabelAgent is not managed by this checkout."; exit 1; }
"$PY" - "$RECORD" <<'PY'
import json, pathlib, sys
record = json.load(open(sys.argv[1], encoding="utf-8"))
ok = True
for row in record.get("processes", []):
    pid, module = int(row["pid"]), row["module"]
    cmd_path = pathlib.Path(f"/proc/{pid}/cmdline")
    alive = cmd_path.exists()
    identity = alive and module in cmd_path.read_bytes().replace(b"\0", b" ").decode(errors="replace")
    print(f"{row['role']}: pid={pid} alive={alive} identity={identity}")
    ok = ok and identity
raise SystemExit(0 if ok else 1)
PY
curl --fail --silent --show-error --max-time 3 http://127.0.0.1:8765/api/run/status >/dev/null
echo "Workbench HTTP read-back: ready"
