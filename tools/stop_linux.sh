#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
RUNTIME="$ROOT/.repro"
RECORD="$RUNTIME/linux-processes.json"
[[ "$(uname -s)" == "Linux" ]] || { echo "Linux only" >&2; exit 2; }
[[ -x "$PY" ]] || { echo "Missing .venv" >&2; exit 2; }
command -v flock >/dev/null || { echo "flock is required" >&2; exit 2; }
mkdir -p "$RUNTIME"
exec 9>"$RUNTIME/linux-lifecycle.lock"
flock -n 9 || { echo "Another LabelAgent lifecycle action is active." >&2; exit 1; }
[[ -f "$RECORD" ]] || { echo "No managed Linux process record."; exit 0; }

mapfile -t rows < <("$PY" - "$RECORD" "$ROOT" <<'PY'
import json, pathlib, sys
record = json.load(open(sys.argv[1], encoding="utf-8"))
if pathlib.Path(record.get("root", "")).resolve() != pathlib.Path(sys.argv[2]).resolve():
    raise SystemExit("Process record belongs to another checkout")
for row in record.get("processes", []):
    print(f"{int(row['pid'])}\t{row['role']}\t{row['module']}")
PY
)
for row in "${rows[@]}"; do
  IFS=$'\t' read -r pid role module <<<"$row"
  [[ -d "/proc/$pid" ]] || continue
  cmdline="$(tr '\0' ' ' <"/proc/$pid/cmdline")"
  if [[ "$cmdline" != *"$module"* ]]; then
    echo "Refusing to stop PID $pid: command identity does not match $module" >&2
    exit 1
  fi
done
for row in "${rows[@]}"; do
  IFS=$'\t' read -r pid _ _ <<<"$row"
  [[ -d "/proc/$pid" ]] && kill -TERM "$pid" 2>/dev/null || true
done
for _ in $(seq 1 50); do
  alive=0
  for row in "${rows[@]}"; do
    IFS=$'\t' read -r pid _ _ <<<"$row"
    [[ -d "/proc/$pid" ]] && alive=1
  done
  ((alive == 0)) && break
  sleep 0.1
done
for row in "${rows[@]}"; do
  IFS=$'\t' read -r pid _ _ <<<"$row"
  [[ -d "/proc/$pid" ]] && kill -KILL "$pid" 2>/dev/null || true
done
rm -f "$RECORD"
echo "Managed LabelAgent Linux processes stopped."
