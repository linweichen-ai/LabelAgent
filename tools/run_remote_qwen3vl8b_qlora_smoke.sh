#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 PROJECT_ROOT OUTPUT_NAME" >&2
  exit 64
fi

PROJECT_ROOT=$1
OUTPUT_NAME=$2
GPU_UUID=GPU-45a88393-aad3-3ee0-022a-4d4dad168088
PYTHON=/home/clw/projects/labelagent-5090/.venv-vlm/bin/python
PEFT_OVERLAY="$PROJECT_ROOT/peft-overlay"
MODEL_DIR=/home/clw/models/labelagent/Qwen3-VL-8B-Instruct
DATASET_DIR="$PROJECT_ROOT/dataset"
OUTPUT_DIR="$PROJECT_ROOT/$OUTPUT_NAME"
VLM_PORT=18878
MINIMUM_FREE_MIB=12000

case "$PROJECT_ROOT" in /home/clw/projects/labelagent-5090/train_runs/qwen3vl8b-qlora-*) ;; *) exit 65 ;; esac
[[ "$OUTPUT_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || exit 66
[[ ! -e "$OUTPUT_DIR" ]] || { echo "training output already exists" >&2; exit 67; }
for required in "$PROJECT_ROOT/train_qwen3vl_lora.py" \
  "$DATASET_DIR/train.jsonl" "$DATASET_DIR/manifest.json" \
  "$MODEL_DIR/config.json" "$PEFT_OVERLAY/peft"; do
  [[ -e "$required" ]] || { echo "missing required path: $required" >&2; exit 68; }
done

exec 9>"$PROJECT_ROOT/.qwen3vl8b-qlora.lock"
flock -n 9 || { echo "another 8B QLoRA experiment is active" >&2; exit 69; }

VLM_HEALTH=$(curl -fsS "http://127.0.0.1:$VLM_PORT/health")
VLM_READY=$(printf '%s' "$VLM_HEALTH" | "$PYTHON" -c \
  'import json,sys; d=json.load(sys.stdin); print(int(d.get("status")=="ok" and d.get("model_loaded") is True and d.get("exact_requests_inflight",-1)==0 and d.get("inference_queue",{}).get("waiting",-1)==0))')
[[ "$VLM_READY" == 1 ]] || { echo "production VLM is not idle and healthy" >&2; exit 70; }
FREE_MIB=$(nvidia-smi --id="$GPU_UUID" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
[[ "$FREE_MIB" =~ ^[0-9]+$ && "$FREE_MIB" -ge "$MINIMUM_FREE_MIB" ]] \
  || { echo "insufficient guarded GPU memory: ${FREE_MIB:-unknown} MiB" >&2; exit 71; }

mkdir -p "$OUTPUT_DIR"
printf '%s\n' "$VLM_HEALTH" >"$OUTPUT_DIR/vlm-health-before.json"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader >"$OUTPUT_DIR/gpu-before.csv"

cd "$PROJECT_ROOT"
set +e
env PYTHONPATH="$PEFT_OVERLAY" CUDA_VISIBLE_DEVICES="$GPU_UUID" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  "$PYTHON" train_qwen3vl_lora.py \
    --model-dir "$MODEL_DIR" --dataset-dir "$DATASET_DIR" --output "$OUTPUT_DIR/run" \
    --max-samples 8 --max-steps 2 --slope-positive-repeat 8 \
    --gradient-accumulation-steps 2 --learning-rate 2e-5 --seed 20261120 \
    --positive-label slope --repeat-evidence present --loss-scope full \
    --quantization nf4 --max-gpu-memory-gib 11 \
    >"$OUTPUT_DIR/train.log" 2>&1
TRAIN_EXIT=$?
set -e

curl -fsS "http://127.0.0.1:$VLM_PORT/health" >"$OUTPUT_DIR/vlm-health-after.json" || true
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader >"$OUTPUT_DIR/gpu-after.csv"
printf '{"training_exit_code":%d,"production_vlm_stopped":false,"test24_used_for_training":false,"production_deployed":false}\n' \
  "$TRAIN_EXIT" >"$OUTPUT_DIR/wrapper-receipt.json"
exit "$TRAIN_EXIT"
