#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
WEB_ONLY=0
WITH_BROWSER_QA=0

usage() {
  cat <<'EOF'
Usage: tools/setup_linux.sh [--python /path/to/python3.12] [--web-only] [--with-browser-qa]

  --web-only        Install workbench dependencies only; use an external SAM3 service.
  --with-browser-qa Install optional Playwright browser acceptance dependencies.
EOF
}

while (($#)); do
  case "$1" in
    --python) PYTHON_BIN="${2:?missing Python path}"; shift 2 ;;
    --web-only) WEB_ONLY=1; shift ;;
    --with-browser-qa) WITH_BROWSER_QA=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$(uname -s)" == "Linux" ]] || {
  echo "This installer is for Linux. Use tools/setup_windows.ps1 on Windows." >&2
  exit 2
}
command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
  echo "Python 3.12 was not found: $PYTHON_BIN" >&2
  echo "Install Python 3.12 or pass --python /absolute/path/to/python3.12." >&2
  exit 2
}
"$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)' || {
  echo "LabelAgent requires Python 3.12 exactly." >&2
  exit 2
}
for command_name in curl flock; do
  command -v "$command_name" >/dev/null 2>&1 || {
    echo "Missing system command: $command_name" >&2
    exit 2
  }
done

cd "$ROOT"
mkdir -p .repro/setup-temp
export TMPDIR="$ROOT/.repro/setup-temp"
export PYTHONUTF8=1
[[ -x .venv/bin/python ]] || "$PYTHON_BIN" -m venv .venv
VENV_PY="$ROOT/.venv/bin/python"
"$VENV_PY" -m pip install --upgrade pip

if ((WEB_ONLY == 0)); then
  "$VENV_PY" -m pip install torch==2.10.0 torchvision==0.25.0 \
    --index-url https://download.pytorch.org/whl/cu128
fi
"$VENV_PY" -m pip install -r requirements.txt
if ((WITH_BROWSER_QA)); then
  "$VENV_PY" -m pip install -r tools/requirements-browser-qa.txt
fi
if ((WEB_ONLY == 0)); then
  "$VENV_PY" -m pip install -e ./source/sam3
  "$VENV_PY" tools/repro_doctor.py --require-cuda
else
  "$VENV_PY" tools/repro_doctor.py
fi

echo "Linux environment ready."
if ((WEB_ONLY)); then
  echo "Start with an external SAM3 service: tools/start_linux.sh --sam-url http://HOST:PORT"
else
  echo "Place the authorized sam3.pt under source/sam3/weights/, then run tools/start_linux.sh"
fi
