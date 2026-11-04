#!/usr/bin/env bash
set -euo pipefail
[[ "$(uname -s)" == Linux ]] || { echo 'Use the PAI Linux Terminal'; exit 1; }
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${LABELAGENT_BLACKWELL_ROOT:?Set this to a NEW project directory on your confirmed data disk}"
PROJECT_ROOT="$LABELAGENT_BLACKWELL_ROOT"
export PIP_NO_CACHE_DIR=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
python3 - "$PROJECT_ROOT" <<'PY'
from pathlib import Path
import shutil,sys
p=Path(sys.argv[1]); assert p.is_absolute(), 'Choose an absolute project directory'
assert not p.exists(), 'Project directory exists; do not rerun blindly or remove it; inspect the prior log'
assert (3,10)<=sys.version_info[:2]<=(3,12), 'Use Python 3.10-3.12'
parent=p.parent
assert parent.is_dir(), 'Choose an existing parent filesystem'
free=shutil.disk_usage(parent).free
print('Available space GiB:',round(free/1024**3,2))
assert free>=50*1024**3, 'Need 50GiB free for the environment and initial small-model test; use a larger project disk'
PY
mkdir -p "$PROJECT_ROOT/logs" "$PROJECT_ROOT/models" "$PROJECT_ROOT/cache/huggingface" "$PROJECT_ROOT/cache/torch" "$PROJECT_ROOT/tmp"
export HF_HOME="$PROJECT_ROOT/cache/huggingface"
export TORCH_HOME="$PROJECT_ROOT/cache/torch"
export XDG_CACHE_HOME="$PROJECT_ROOT/cache"
export TMPDIR="$PROJECT_ROOT/tmp"
python3 - "$PROJECT_ROOT" <<'PY'
from pathlib import Path
import shlex,sys
r=Path(sys.argv[1])
values={'LABELAGENT_BLACKWELL_ROOT':str(r),'HF_HOME':str(r/'cache/huggingface'),
        'TORCH_HOME':str(r/'cache/torch'),'XDG_CACHE_HOME':str(r/'cache'),
        'TMPDIR':str(r/'tmp'),'PIP_NO_CACHE_DIR':'1'}
with (r/'env-blackwell.sh').open('x',encoding='utf-8') as f:
    f.write(''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in values.items()))
PY
python3 -m venv "$PROJECT_ROOT/venv-sdpa"
PY="$PROJECT_ROOT/venv-sdpa/bin/python"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
"$PY" -m pip install transformers==4.57.6 peft==0.17.1 accelerate==1.10.1 \
  Pillow==11.3.0 numpy==2.2.6 qwen-vl-utils==0.0.14
"$PY" -m pip check
"$PY" -m pip freeze > "$PROJECT_ROOT/logs/installed-sdpa.txt"
"$PY" "$SCRIPT_DIR/preflight_blackwell_sdpa.py" --out "$PROJECT_ROOT/logs/preflight-blackwell-sdpa.json"
printf '\nEnvironment and primitive GPU checks passed. Model weights and training have not started.\n'
printf 'Python: %s\n' "$PY"
printf 'Receipt: %s/logs/preflight-blackwell-sdpa.json\n' "$PROJECT_ROOT"
