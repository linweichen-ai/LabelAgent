#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 7 ]]; then
  echo "usage: $0 PROJECT_ROOT OUTPUT_NAME [MAX_SAMPLES] [MAX_STEPS] [SLOPE_REPEAT] [LEARNING_RATE] [SEED]" >&2
  exit 64
fi

PROJECT_ROOT=$1
OUTPUT_NAME=$2
MAX_SAMPLES=${3:-104}
MAX_STEPS=${4:-52}
SLOPE_REPEAT=${5:-8}
LEARNING_RATE=${6:-1e-4}
SEED=${7:-20261118}
GPU_UUID=GPU-8dde4d59-7e14-09ff-46c0-35da2f7177b8
SAM_PORT=18766
VLM_PORT=18878
MINIMUM_FREE_MIB=12000
PYTHON=/home/clw/labelagent_vlm_20261104/venv/bin/python
PEFT_OVERLAY="$PROJECT_ROOT/venv/lib/python3.12/site-packages"
MODEL_DIR=/home/clw/labelagent_vlm_20261104/models/Qwen3-VL-2B-Instruct
OUTPUT_DIR="$PROJECT_ROOT/$OUTPUT_NAME"

case "$PROJECT_ROOT" in /home/clw/projects/sfl-cpd15y/*) ;; *) exit 65 ;; esac
[[ "$OUTPUT_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || exit 66
[[ "$MAX_SAMPLES" =~ ^[0-9]+$ && "$MAX_SAMPLES" -ge 1 && "$MAX_SAMPLES" -le 128 ]] || exit 67
[[ "$MAX_STEPS" =~ ^[0-9]+$ && "$MAX_STEPS" -ge 1 && "$MAX_STEPS" -le 100 ]] || exit 68
[[ "$SLOPE_REPEAT" =~ ^[1-8]$ ]] || exit 69
[[ "$LEARNING_RATE" =~ ^(0\.[0-9]+|[0-9]+(\.[0-9]+)?e-[0-9]+)$ ]] || exit 70
[[ "$SEED" =~ ^[0-9]+$ && "$SEED" -le 2147483647 ]] || exit 71
[[ ! -e "$OUTPUT_DIR" ]] || { echo "training output already exists" >&2; exit 72; }

for required in "$PROJECT_ROOT/train_qwen3vl_lora.py" \
  "$PROJECT_ROOT/dataset/train.jsonl" "$PROJECT_ROOT/dataset/manifest.json" \
  "$MODEL_DIR/config.json" "$PEFT_OVERLAY"; do
  [[ -e "$required" ]] || { echo "missing required path: $required" >&2; exit 73; }
done

exec 9>"$PROJECT_ROOT/.qwen3vl2b-parallel.lock"
flock -n 9 || { echo "another parallel 2B experiment is active" >&2; exit 74; }

SAM_PID=$(pgrep -f "sam_service_fastapi.py.*--port $SAM_PORT" | head -n 1)
[[ -n "$SAM_PID" && $(ps -p "$SAM_PID" -o user= | xargs) == clw ]] || exit 75
SAM_HEALTH=$(curl -fsS "http://127.0.0.1:$SAM_PORT/health")
SAM_INFLIGHT=$(printf '%s' "$SAM_HEALTH" | "$PYTHON" -c \
  'import json,sys; print(json.load(sys.stdin).get("response_cache",{}).get("in_flight",-1))')
[[ "$SAM_INFLIGHT" == 0 ]] || { echo "SAM has in-flight requests: $SAM_INFLIGHT" >&2; exit 76; }
VLM_HEALTH=$(curl -fsS "http://127.0.0.1:$VLM_PORT/health")
VLM_READY=$(printf '%s' "$VLM_HEALTH" | "$PYTHON" -c \
  'import json,sys; d=json.load(sys.stdin); print(int(d.get("status")=="ok" and d.get("model_loaded") is True))')
[[ "$VLM_READY" == 1 ]] || { echo "production VLM is not healthy" >&2; exit 77; }
FREE_MIB=$(nvidia-smi --id="$GPU_UUID" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
[[ "$FREE_MIB" =~ ^[0-9]+$ && "$FREE_MIB" -ge "$MINIMUM_FREE_MIB" ]] \
  || { echo "insufficient shared SAM GPU memory: ${FREE_MIB:-unknown} MiB" >&2; exit 78; }

cd "$PROJECT_ROOT"
env PYTHONPATH="$PEFT_OVERLAY" CUDA_VISIBLE_DEVICES="$GPU_UUID" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  "$PYTHON" train_qwen3vl_lora.py \
    --model-dir "$MODEL_DIR" --dataset-dir dataset --output "$OUTPUT_NAME" \
    --max-samples "$MAX_SAMPLES" --max-steps "$MAX_STEPS" \
    --slope-positive-repeat "$SLOPE_REPEAT" --gradient-accumulation-steps 2 \
    --learning-rate "$LEARNING_RATE" --seed "$SEED"

curl -fsS "http://127.0.0.1:$SAM_PORT/health" >"$OUTPUT_DIR/sam-health-after.json"
curl -fsS "http://127.0.0.1:$VLM_PORT/health" >"$OUTPUT_DIR/vlm-health-after.json"
echo "parallel_training_complete=true"

