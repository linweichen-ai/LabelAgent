#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 PROJECT_ROOT OUTPUT_NAME [smoke|candidate8|candidate16]" >&2
  exit 64
fi

PROJECT_ROOT=$1
OUTPUT_NAME=$2
PROFILE=${3:-smoke}
GPU_UUID=GPU-8dde4d59-7e14-09ff-46c0-35da2f7177b8
PYTHON=/home/clw/labelagent_vlm_20261104/venv/bin/python
PEFT_OVERLAY=/home/clw/projects/sfl-cpd15y/qwen3vl2b-lora-round102/venv/lib/python3.12/site-packages
MODEL_DIR=/home/clw/labelagent_vlm_20261104/models/Qwen3-VL-8B-Instruct
DATASET_DIR="$PROJECT_ROOT/dataset"
OUTPUT_DIR="$PROJECT_ROOT/$OUTPUT_NAME"
SAM_PORT=18766
VLM_PORT=18878
MINIMUM_FREE_MIB=16000

case "$PROJECT_ROOT" in /home/clw/projects/sfl-cpd15y/qwen3vl8b-qlora-*) ;; *) exit 65 ;; esac
[[ "$OUTPUT_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || exit 66
[[ "$PROFILE" =~ ^(smoke|candidate8|candidate16)$ ]] || exit 73
[[ ! -e "$OUTPUT_DIR" ]] || { echo "training output already exists" >&2; exit 67; }
for required in "$PROJECT_ROOT/train_qwen3vl_lora.py" \
  "$DATASET_DIR/train.jsonl" "$DATASET_DIR/manifest.json" \
  "$MODEL_DIR/config.json" "$PEFT_OVERLAY/peft"; do
  [[ -e "$required" ]] || { echo "missing required path: $required" >&2; exit 68; }
done

exec 9>"$PROJECT_ROOT/.qwen3vl8b-qlora.lock"
flock -n 9 || { echo "another 8B QLoRA experiment is active" >&2; exit 69; }

SAM_HEALTH=$(curl -fsS "http://127.0.0.1:$SAM_PORT/health")
SAM_INFLIGHT=$(printf '%s' "$SAM_HEALTH" | "$PYTHON" -c \
  'import json,sys; print(json.load(sys.stdin).get("response_cache",{}).get("in_flight",-1))')
[[ "$SAM_INFLIGHT" == 0 ]] || { echo "SAM has in-flight requests" >&2; exit 70; }
VLM_HEALTH=$(curl -fsS "http://127.0.0.1:$VLM_PORT/health")
VLM_READY=$(printf '%s' "$VLM_HEALTH" | "$PYTHON" -c \
  'import json,sys; d=json.load(sys.stdin); print(int(d.get("status")=="ok" and d.get("model_loaded") is True and d.get("exact_requests_inflight",-1)==0))')
[[ "$VLM_READY" == 1 ]] || { echo "production VLM is not idle and healthy" >&2; exit 71; }
FREE_MIB=$(nvidia-smi --id="$GPU_UUID" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
[[ "$FREE_MIB" =~ ^[0-9]+$ && "$FREE_MIB" -ge "$MINIMUM_FREE_MIB" ]] \
  || { echo "insufficient guarded GPU memory: ${FREE_MIB:-unknown} MiB" >&2; exit 72; }

if [[ "$PROFILE" == candidate16 ]]; then
  MAX_SAMPLES=104
  MAX_STEPS=16
  LEARNING_RATE=2e-5
elif [[ "$PROFILE" == candidate8 ]]; then
  MAX_SAMPLES=104
  MAX_STEPS=8
  LEARNING_RATE=1e-5
else
  MAX_SAMPLES=8
  MAX_STEPS=2
  LEARNING_RATE=2e-5
fi

mkdir -p "$OUTPUT_DIR"
printf '%s\n' "$SAM_HEALTH" >"$OUTPUT_DIR/sam-health-before.json"
printf '%s\n' "$VLM_HEALTH" >"$OUTPUT_DIR/vlm-health-before.json"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader >"$OUTPUT_DIR/gpu-before.csv"

cd "$PROJECT_ROOT"
set +e
env PYTHONPATH="$PEFT_OVERLAY" CUDA_VISIBLE_DEVICES="$GPU_UUID" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  "$PYTHON" train_qwen3vl_lora.py \
    --model-dir "$MODEL_DIR" --dataset-dir "$DATASET_DIR" --output "$OUTPUT_DIR/run" \
    --max-samples "$MAX_SAMPLES" --max-steps "$MAX_STEPS" --slope-positive-repeat 8 \
    --gradient-accumulation-steps 2 --learning-rate "$LEARNING_RATE" --seed 20261120 \
    --positive-label slope --repeat-evidence present --loss-scope full \
    --quantization nf4 --max-gpu-memory-gib 13 \
    >"$OUTPUT_DIR/train.log" 2>&1
TRAIN_EXIT=$?
set -e

curl -fsS "http://127.0.0.1:$SAM_PORT/health" >"$OUTPUT_DIR/sam-health-after.json" || true
curl -fsS "http://127.0.0.1:$VLM_PORT/health" >"$OUTPUT_DIR/vlm-health-after.json" || true
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader >"$OUTPUT_DIR/gpu-after.csv"
printf '{"training_exit_code":%d,"services_stopped":false,"test24_used_for_training":false,"production_deployed":false}\n' \
  "$TRAIN_EXIT" >"$OUTPUT_DIR/wrapper-receipt.json"
exit "$TRAIN_EXIT"
