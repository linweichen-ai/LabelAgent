#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
RUNTIME="$ROOT/.repro"
LOGS="$RUNTIME/logs"
RECORD="$RUNTIME/linux-processes.json"
SAM_URL="http://127.0.0.1:8766"
WORK_DIR="$ROOT/work"
HOST="127.0.0.1"
NO_BROWSER=0

usage() {
  cat <<'EOF'
Usage: tools/start_linux.sh [--sam-url URL] [--work-dir DIR] [--host ADDRESS] [--no-browser]

Default binding is 127.0.0.1. For remote use, keep it and create an SSH tunnel:
  ssh -L 8765:127.0.0.1:8765 USER@LINUX_HOST
EOF
}

while (($#)); do
  case "$1" in
    --sam-url) SAM_URL="${2:?missing URL}"; shift 2 ;;
    --work-dir) WORK_DIR="${2:?missing work directory}"; shift 2 ;;
    --host) HOST="${2:?missing bind address}"; shift 2 ;;
    --no-browser) NO_BROWSER=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$(uname -s)" == "Linux" ]] || { echo "Linux only" >&2; exit 2; }
[[ -x "$PY" ]] || { echo "Missing .venv. Run tools/setup_linux.sh first." >&2; exit 2; }
for command_name in curl flock; do
  command -v "$command_name" >/dev/null 2>&1 || { echo "$command_name is required" >&2; exit 2; }
done

mkdir -p "$LOGS" "$RUNTIME/temp" "$WORK_DIR"
exec 9>"$RUNTIME/linux-lifecycle.lock"
flock -n 9 || { echo "Another LabelAgent lifecycle action is active." >&2; exit 1; }

if [[ -f "$RECORD" ]]; then
  if "$PY" - "$RECORD" <<'PY'
import json, os, sys
record = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if any(os.path.exists(f"/proc/{row['pid']}") for row in record.get("processes", [])) else 1)
PY
  then
    echo "A managed LabelAgent process is already alive. Run tools/status_linux.sh." >&2
    exit 1
  fi
  mv "$RECORD" "$RUNTIME/linux-processes.stale.$(date -u +%Y%m%dT%H%M%SZ).json"
fi

export PYTHONUTF8=1
export TMPDIR="$RUNTIME/temp"
export LABELAGENT_WORK_DIR="$(cd "$WORK_DIR" && pwd)"
export SAM3_URL="$SAM_URL"
export AUTOLABEL_HOST="$HOST"
export AUTOLABEL_PORT=8765
sam_pid=""
web_pid=""
cleanup_failed_start() {
  local code=$?
  if ((code != 0)); then
    [[ -n "$web_pid" ]] && kill "$web_pid" 2>/dev/null || true
    [[ -n "$sam_pid" ]] && kill "$sam_pid" 2>/dev/null || true
    rm -f "$RECORD"
  fi
}
trap cleanup_failed_start EXIT

cd "$ROOT"
if [[ "$SAM_URL" == "http://127.0.0.1:8766" ]]; then
  "$PY" tools/repro_doctor.py --require-web-port --require-sam-port --require-cuda --require-weight
  nohup "$PY" -m source.sam3_service.server >"$LOGS/sam3.out.log" 2>"$LOGS/sam3.err.log" &
  sam_pid=$!
else
  curl --fail --silent --show-error --max-time 5 "${SAM_URL%/}/health" >/dev/null
  "$PY" tools/repro_doctor.py --require-web-port
fi

nohup "$PY" -m source.agent.server >"$LOGS/workbench.out.log" 2>"$LOGS/workbench.err.log" &
web_pid=$!
"$PY" - "$RECORD" "$ROOT" "$SAM_URL" "$sam_pid" "$web_pid" <<'PY'
import datetime, json, os, pathlib, sys, tempfile
record_path, root, sam_url, sam_pid, web_pid = sys.argv[1:]
processes = [{"pid": int(web_pid), "role": "workbench", "module": "source.agent.server"}]
if sam_pid:
    processes.insert(0, {"pid": int(sam_pid), "role": "sam3", "module": "source.sam3_service.server"})
payload = {"schema_version": "labelagent.linux-process-record.v1",
           "root": str(pathlib.Path(root).resolve()), "sam_url": sam_url,
           "python": str(pathlib.Path(sys.executable).resolve()),
           "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "processes": processes}
target = pathlib.Path(record_path)
with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as stream:
    json.dump(payload, stream, ensure_ascii=False, indent=2)
    temporary = stream.name
os.replace(temporary, target)
PY

ready=0
for _ in $(seq 1 60); do
  if ! kill -0 "$web_pid" 2>/dev/null; then break; fi
  if curl --fail --silent --max-time 2 http://127.0.0.1:8765/api/run/status >/dev/null; then
    ready=1
    break
  fi
  sleep 0.5
done
if ((ready == 0)); then
  echo "Workbench did not become ready. See $LOGS/workbench.err.log" >&2
  exit 1
fi
"$PY" tools/repro_doctor.py --require-services --sam-url "$SAM_URL"
trap - EXIT
if ((NO_BROWSER == 0)) && [[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]] && command -v xdg-open >/dev/null; then
  xdg-open http://127.0.0.1:8765 >/dev/null 2>&1 || true
fi
echo "LabelAgent ready: http://127.0.0.1:8765"
echo "Logs: $LOGS"
