#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 6 ]]; then
  echo "usage: $0 PROJECT_ROOT OUTPUT_NAME ADAPTER_RUN_NAME [BASE_RESULT_NAME] [SPLIT] [DATASET_NAME]" >&2
  exit 64
fi
PROJECT_ROOT=$1
OUTPUT_NAME=$2
ADAPTER_RUN_NAME=$3
BASE_RESULT_NAME=${4:-}
SPLIT=${5:-dev}
DATASET_NAME=${6:-dataset}
GPU_UUID=GPU-8dde4d59-7e14-09ff-46c0-35da2f7177b8
SAM_PORT=18766
VLM_PORT=18878
MINIMUM_FREE_MIB=12000
PYTHON=/home/clw/labelagent_vlm_20261104/venv/bin/python
PEFT_OVERLAY="$PROJECT_ROOT/venv/lib/python3.12/site-packages"
MODEL_DIR=/home/clw/labelagent_vlm_20261104/models/Qwen3-VL-2B-Instruct
ADAPTER_DIR="$PROJECT_ROOT/$ADAPTER_RUN_NAME/adapter"
SERVER_DIR=/home/clw/labelagent_remote_20261110/vlm
OUTPUT_DIR="$PROJECT_ROOT/$OUTPUT_NAME"

case "$PROJECT_ROOT" in /home/clw/projects/sfl-cpd15y/*) ;; *) exit 65 ;; esac
[[ "$OUTPUT_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || exit 66
[[ "$ADAPTER_RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || exit 67
[[ -z "$BASE_RESULT_NAME" || "$BASE_RESULT_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || exit 74
[[ "$SPLIT" =~ ^(dev|holdout)$ ]] || exit 76
[[ "$DATASET_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || exit 77
[[ ! -e "$OUTPUT_DIR" ]] || { echo "evaluation output already exists" >&2; exit 68; }
for required in "$PROJECT_ROOT/evaluate_qwen3vl_lora_generation.py" \
  "$PROJECT_ROOT/evaluate_qwen3vl_lora_teacher_forcing.py" \
  "$PROJECT_ROOT/train_qwen3vl_lora.py" "$PROJECT_ROOT/$DATASET_NAME/dev.jsonl" \
  "$PROJECT_ROOT/$DATASET_NAME/manifest.json" "$PROJECT_ROOT/$DATASET_NAME/contract.json" \
  "$ADAPTER_DIR/adapter_model.safetensors" "$MODEL_DIR/config.json" \
  "$SERVER_DIR/local_vlm_server.py" "$PEFT_OVERLAY"; do
  [[ -e "$required" ]] || { echo "missing required path: $required" >&2; exit 69; }
done

exec 9>"$PROJECT_ROOT/.qwen3vl2b-parallel.lock"
flock -n 9 || { echo "another parallel 2B experiment is active" >&2; exit 70; }
SAM_HEALTH=$(curl -fsS "http://127.0.0.1:$SAM_PORT/health")
SAM_INFLIGHT=$(printf '%s' "$SAM_HEALTH" | "$PYTHON" -c \
  'import json,sys; print(json.load(sys.stdin).get("response_cache",{}).get("in_flight",-1))')
[[ "$SAM_INFLIGHT" == 0 ]] || { echo "SAM has in-flight requests: $SAM_INFLIGHT" >&2; exit 71; }
VLM_HEALTH=$(curl -fsS "http://127.0.0.1:$VLM_PORT/health")
VLM_READY=$(printf '%s' "$VLM_HEALTH" | "$PYTHON" -c \
  'import json,sys; d=json.load(sys.stdin); print(int(d.get("status")=="ok" and d.get("model_loaded") is True))')
[[ "$VLM_READY" == 1 ]] || { echo "production VLM is not healthy" >&2; exit 72; }
FREE_MIB=$(nvidia-smi --id="$GPU_UUID" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
[[ "$FREE_MIB" =~ ^[0-9]+$ && "$FREE_MIB" -ge "$MINIMUM_FREE_MIB" ]] \
  || { echo "insufficient shared SAM GPU memory: ${FREE_MIB:-unknown} MiB" >&2; exit 73; }

cd "$PROJECT_ROOT"
BASE_ARGS=()
if [[ -n "$BASE_RESULT_NAME" ]]; then
  BASE_PATH="$PROJECT_ROOT/$BASE_RESULT_NAME/result.json"
  [[ -f "$BASE_PATH" ]] || { echo "missing base result: $BASE_PATH" >&2; exit 75; }
  BASE_ARGS=(--base-result "$BASE_PATH")
fi
env PYTHONPATH="$PEFT_OVERLAY:$PROJECT_ROOT:$SERVER_DIR" \
  CUDA_VISIBLE_DEVICES="$GPU_UUID" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false "$PYTHON" evaluate_qwen3vl_lora_generation.py \
    --model-dir "$MODEL_DIR" --adapter-dir "$ADAPTER_DIR" \
    --dataset-dir "$DATASET_NAME" --contract "$DATASET_NAME/contract.json" \
    --server-module-dir "$SERVER_DIR" --output "$OUTPUT_NAME" \
    --split "$SPLIT" \
    "${BASE_ARGS[@]}"

curl -fsS "http://127.0.0.1:$SAM_PORT/health" >"$OUTPUT_DIR/sam-health-after.json"
curl -fsS "http://127.0.0.1:$VLM_PORT/health" >"$OUTPUT_DIR/vlm-health-after.json"
echo "parallel_generation_evaluation_complete=true"
