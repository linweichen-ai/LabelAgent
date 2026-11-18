#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 PROJECT_ROOT OUTPUT_NAME ADAPTER_RUN_NAME MANIFEST_NAME" >&2
  exit 64
fi
PROJECT_ROOT=$1
OUTPUT_NAME=$2
ADAPTER_RUN_NAME=$3
MANIFEST_NAME=$4
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
MANIFEST_PATH="$PROJECT_ROOT/$MANIFEST_NAME"

case "$PROJECT_ROOT" in /home/clw/projects/sfl-cpd15y/*) ;; *) exit 65 ;; esac
for value in "$OUTPUT_NAME" "$ADAPTER_RUN_NAME" "$MANIFEST_NAME"; do
  [[ "$value" =~ ^[A-Za-z0-9._-]+$ ]] || exit 66
done
[[ ! -e "$OUTPUT_DIR" ]] || { echo "shadow output already exists" >&2; exit 67; }
for required in "$PROJECT_ROOT/run_qwen3vl_lora_shadow.py" \
  "$PROJECT_ROOT/dataset/contract.json" "$MANIFEST_PATH" \
  "$ADAPTER_DIR/adapter_model.safetensors" "$MODEL_DIR/config.json" \
  "$SERVER_DIR/local_vlm_server.py" "$PEFT_OVERLAY"; do
  [[ -e "$required" ]] || { echo "missing required path: $required" >&2; exit 68; }
done
exec 9>"$PROJECT_ROOT/.qwen3vl2b-parallel.lock"
flock -n 9 || { echo "another parallel 2B experiment is active" >&2; exit 69; }
SAM_HEALTH=$(curl -fsS "http://127.0.0.1:$SAM_PORT/health")
SAM_INFLIGHT=$(printf '%s' "$SAM_HEALTH" | "$PYTHON" -c \
  'import json,sys; print(json.load(sys.stdin).get("response_cache",{}).get("in_flight",-1))')
[[ "$SAM_INFLIGHT" == 0 ]] || exit 70
VLM_HEALTH=$(curl -fsS "http://127.0.0.1:$VLM_PORT/health")
VLM_READY=$(printf '%s' "$VLM_HEALTH" | "$PYTHON" -c \
  'import json,sys; d=json.load(sys.stdin); print(int(d.get("status")=="ok" and d.get("model_loaded") is True))')
[[ "$VLM_READY" == 1 ]] || exit 71
FREE_MIB=$(nvidia-smi --id="$GPU_UUID" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
[[ "$FREE_MIB" =~ ^[0-9]+$ && "$FREE_MIB" -ge "$MINIMUM_FREE_MIB" ]] || exit 72

cd "$PROJECT_ROOT"
env PYTHONPATH="$PEFT_OVERLAY:$PROJECT_ROOT:$SERVER_DIR" \
  CUDA_VISIBLE_DEVICES="$GPU_UUID" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false "$PYTHON" run_qwen3vl_lora_shadow.py \
    --model-dir "$MODEL_DIR" --adapter-dir "$ADAPTER_DIR" \
    --contract dataset/contract.json --manifest "$MANIFEST_PATH" \
    --server-module-dir "$SERVER_DIR" --output "$OUTPUT_NAME"
curl -fsS "http://127.0.0.1:$SAM_PORT/health" >"$OUTPUT_DIR/sam-health-after.json"
curl -fsS "http://127.0.0.1:$VLM_PORT/health" >"$OUTPUT_DIR/vlm-health-after.json"
echo "parallel_shadow_complete=true"
