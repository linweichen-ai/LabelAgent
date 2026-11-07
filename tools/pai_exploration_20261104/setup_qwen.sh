#!/usr/bin/env bash
set -euo pipefail
[[ "$(uname -s)" == Linux ]] || { echo 'Run in Linux PAI DSW'; exit 1; }
export LABELAGENT_PAI_ROOT="${LABELAGENT_PAI_ROOT:-/mnt/workspace/labelagent-pai-exploration-20261104}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python3 -c 'import sys; assert (3,10)<=sys.version_info[:2]<=(3,12), "Choose Python 3.10-3.12 GPU image"'
[[ ! -e "$LABELAGENT_PAI_ROOT" ]] || { echo 'Choose a fresh LABELAGENT_PAI_ROOT; existing directory preserved'; exit 1; }
mkdir -p "$LABELAGENT_PAI_ROOT/logs" "$LABELAGENT_PAI_ROOT/models"
python3 -m venv "$LABELAGENT_PAI_ROOT/venv-qwen"
PY="$LABELAGENT_PAI_ROOT/venv-qwen/bin/python"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
"$PY" -m pip install -r "$SCRIPT_DIR/requirements-qwen.txt"
MAX_JOBS=4 "$PY" -m pip install flash-attn==2.8.3 --no-build-isolation
PIN="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["qwen_git_commit"])' "$SCRIPT_DIR/upstream_pin.json")"
git clone --filter=blob:none https://github.com/QwenLM/Qwen3-VL.git "$LABELAGENT_PAI_ROOT/Qwen3-VL"
git -C "$LABELAGENT_PAI_ROOT/Qwen3-VL" checkout --detach "$PIN"
"$PY" -m pip check
"$PY" -m pip freeze > "$LABELAGENT_PAI_ROOT/logs/qwen-installed.txt"
"$PY" "$SCRIPT_DIR/preflight.py" --out "$LABELAGENT_PAI_ROOT/logs/preflight.json"
echo 'Training environment installed; model weights and training have not started.'
