#!/usr/bin/env bash
set -euo pipefail
[[ "$(uname -s)" == Linux ]] || { echo 'Run inside the Linux PAI DSW terminal.' >&2; exit 1; }
PROJECT_ROOT="${LABELAGENT_PAI_ROOT:-/mnt/workspace/labelagent-pai-vlm32b}"
mkdir -p "$PROJECT_ROOT/logs" "$PROJECT_ROOT/cache"
python3 -c 'import sys; assert sys.version_info >= (3,11), "Use a Python 3.11+ GPU image"'
nvidia-smi
if [[ -e "$PROJECT_ROOT/.venv" ]]; then
  echo 'Existing project environment found. Reuse it or choose a fresh LABELAGENT_PAI_ROOT; it was not changed.' >&2
  exit 1
fi
python3 -m venv "$PROJECT_ROOT/.bootstrap"
"$PROJECT_ROOT/.bootstrap/bin/python" -m pip install --upgrade pip uv
UV="$PROJECT_ROOT/.bootstrap/bin/uv"
"$UV" venv --seed --python "$(command -v python3)" "$PROJECT_ROOT/.venv"
"$UV" pip install --python "$PROJECT_ROOT/.venv/bin/python" \
  vllm modelscope --torch-backend=auto --only-binary=vllm,torch,torchvision
"$UV" pip freeze --python "$PROJECT_ROOT/.venv/bin/python" > "$PROJECT_ROOT/logs/installed-packages.txt"
"$PROJECT_ROOT/.venv/bin/python" -m pip check
printf 'Environment ready: %s\n' "$PROJECT_ROOT/.venv"
printf 'Next: activate this environment and download the model. No weights were downloaded by this script.\n'
