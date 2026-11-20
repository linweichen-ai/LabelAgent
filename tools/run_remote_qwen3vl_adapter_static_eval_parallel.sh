#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 5 ]]; then
  echo "usage: $0 PROJECT_ROOT OUTPUT_NAME ADAPTER_RUN_NAME [DATASET_NAME] [REFERENCE_RESULT_NAME]" >&2
  exit 64
fi
PROJECT_ROOT=$1
OUTPUT_NAME=$2
ADAPTER_RUN_NAME=$3
DATASET_NAME=${4:-dataset}
REFERENCE_RESULT_NAME=${5:-generation-eval-train1epoch-production-order-v1}
GPU_UUID=GPU-8dde4d59-7e14-09ff-46c0-35da2f7177b8
SAM_PORT=18766
VLM_PORT=18878
MINIMUM_FREE_MIB=12000
PYTHON=/home/clw/labelagent_vlm_20261104/venv/bin/python
PEFT_OVERLAY="$PROJECT_ROOT/venv/lib/python3.12/site-packages"
SERVER_DIR=/home/clw/labelagent_remote_20261110/vlm
MODEL_DIR=/home/clw/labelagent_vlm_20261104/models/Qwen3-VL-2B-Instruct
ADAPTER_DIR="$PROJECT_ROOT/$ADAPTER_RUN_NAME/adapter"
OUTPUT_DIR="$PROJECT_ROOT/$OUTPUT_NAME"
REFERENCE_RESULT="$PROJECT_ROOT/$REFERENCE_RESULT_NAME/result.json"
TRITON_CACHE_DIR="$PROJECT_ROOT/cache/triton-qwen3vl2b-parallel"
TORCHINDUCTOR_CACHE_DIR="$PROJECT_ROOT/cache/torchinductor-qwen3vl2b-parallel"

case "$PROJECT_ROOT" in /home/clw/projects/sfl-cpd15y/*) ;; *) exit 65 ;; esac
[[ "$OUTPUT_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || exit 66
[[ "$ADAPTER_RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || exit 67
[[ "$DATASET_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || exit 73
[[ "$REFERENCE_RESULT_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || exit 74
[[ ! -e "$OUTPUT_DIR" ]] || { echo "evaluation output already exists" >&2; exit 68; }
for required in "$PROJECT_ROOT/evaluate_qwen3vl_merged_generation.py" \
  "$PROJECT_ROOT/train_qwen3vl_lora.py" "$PROJECT_ROOT/$DATASET_NAME/dev.jsonl" \
  "$PROJECT_ROOT/$DATASET_NAME/manifest.json" "$PROJECT_ROOT/$DATASET_NAME/contract.json" \
  "$MODEL_DIR/config.json" "$ADAPTER_DIR/adapter_model.safetensors" \
  "$REFERENCE_RESULT" "$SERVER_DIR/local_vlm_server.py" "$PEFT_OVERLAY"; do
  [[ -e "$required" ]] || { echo "missing required path: $required" >&2; exit 69; }
done

exec 9>"$PROJECT_ROOT/.qwen3vl2b-parallel.lock"
flock -n 9 || { echo "another parallel 2B experiment is active" >&2; exit 70; }
SAM_HEALTH=$(curl -fsS "http://127.0.0.1:$SAM_PORT/health")
SAM_INFLIGHT=$(printf '%s' "$SAM_HEALTH" | "$PYTHON" -c \
  'import json,sys; print(json.load(sys.stdin).get("response_cache",{}).get("in_flight",-1))')
[[ "$SAM_INFLIGHT" == 0 ]] || { echo "SAM has in-flight requests: $SAM_INFLIGHT" >&2; exit 71; }
curl -fsS "http://127.0.0.1:$VLM_PORT/health" >/dev/null
FREE_MIB=$(nvidia-smi --id="$GPU_UUID" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
[[ "$FREE_MIB" =~ ^[0-9]+$ && "$FREE_MIB" -ge "$MINIMUM_FREE_MIB" ]] \
  || { echo "insufficient shared SAM GPU memory: ${FREE_MIB:-unknown} MiB" >&2; exit 72; }
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

cd "$PROJECT_ROOT"
env PYTHONPATH="$PEFT_OVERLAY:$PROJECT_ROOT:$SERVER_DIR" \
  CUDA_VISIBLE_DEVICES="$GPU_UUID" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  TORCHINDUCTOR_CACHE_DIR="$TORCHINDUCTOR_CACHE_DIR" \
  "$PYTHON" evaluate_qwen3vl_merged_generation.py \
    --model-dir "$MODEL_DIR" --adapter-dir "$ADAPTER_DIR" \
    --dataset-dir "$DATASET_NAME" --contract "$DATASET_NAME/contract.json" \
    --server-module-dir "$SERVER_DIR" --reference-result "$REFERENCE_RESULT" \
    --cache-implementation static --output "$OUTPUT_NAME"

curl -fsS "http://127.0.0.1:$SAM_PORT/health" >"$OUTPUT_DIR/sam-health-after.json"
curl -fsS "http://127.0.0.1:$VLM_PORT/health" >"$OUTPUT_DIR/vlm-health-after.json"
echo "parallel_static_eval_complete=true"

