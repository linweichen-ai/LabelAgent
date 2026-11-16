#!/usr/bin/env bash
set -euo pipefail
[[ "$(uname -s)" == Linux ]] || { echo 'Run inside PAI DSW.' >&2; exit 1; }
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${LABELAGENT_PAI_ROOT:-/mnt/workspace/labelagent-pai-vlm32b}"
MODEL_DIR="${LABELAGENT_MODEL_DIR:-$PROJECT_ROOT/models/Qwen3-VL-32B-Instruct}"
TP="${LABELAGENT_TP:-2}"
PIXELS="${LABELAGENT_MAX_PIXELS:-262144}"
source "$PROJECT_ROOT/.venv/bin/activate"
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false VLLM_NO_USAGE_STATS=1
# Keep PAI's allocated CUDA_VISIBLE_DEVICES unchanged.
CACHE_ROOT="${LABELAGENT_RUNTIME_CACHE:-/tmp/labelagent-vlm32b-runtime-${UID}}"
mkdir -p "$CACHE_ROOT" "$PROJECT_ROOT/logs"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/inductor"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export CUDA_CACHE_PATH="$CACHE_ROOT/cuda"
export VLLM_CACHE_ROOT="$CACHE_ROOT/vllm"
if [[ -z "${VLLM_API_KEY:-}" ]]; then
  read -r -s -p 'Set your private VLM API key (use the same value in LabelAgent): ' VLLM_API_KEY
  printf '\n'
  [[ -n "$VLLM_API_KEY" ]] || { echo 'A nonempty API key is required.' >&2; exit 1; }
  export VLLM_API_KEY
fi
python "$SCRIPT_DIR/preflight_pai.py" --expected-gpus "$TP" --model-dir "$MODEL_DIR" \
  --output "$PROJECT_ROOT/logs/preflight-latest.json"
exec vllm serve "$MODEL_DIR" \
  --served-model-name labelagent-vlm32b \
  --host 127.0.0.1 --port 18878 \
  --tensor-parallel-size "$TP" --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 16384 --max-num-seqs 1 \
  --limit-mm-per-prompt '{"image":8,"video":0}' \
  --mm-processor-kwargs "{\"min_pixels\":65536,\"max_pixels\":${PIXELS}}" \
  --generation-config vllm --seed 0 --enforce-eager \
  --disable-custom-all-reduce
