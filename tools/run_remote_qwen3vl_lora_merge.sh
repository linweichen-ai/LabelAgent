#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 PROJECT_ROOT OUTPUT_NAME ADAPTER_RUN_NAME" >&2
  exit 64
fi
PROJECT_ROOT=$1
OUTPUT_NAME=$2
ADAPTER_RUN_NAME=$3
PYTHON=/home/clw/labelagent_vlm_20261104/venv/bin/python
PEFT_OVERLAY="$PROJECT_ROOT/venv/lib/python3.12/site-packages"
MODEL_DIR=/home/clw/labelagent_vlm_20261104/models/Qwen3-VL-2B-Instruct
ADAPTER_DIR="$PROJECT_ROOT/$ADAPTER_RUN_NAME/adapter"
OUTPUT_DIR="$PROJECT_ROOT/$OUTPUT_NAME"
EXPECTED_ADAPTER_SHA256=112cafc481d7d1d4c7e9103f09fc964798881448a3efe3fa277400b929a814ef

case "$PROJECT_ROOT" in /home/clw/projects/sfl-cpd15y/*) ;; *) exit 65 ;; esac
[[ "$OUTPUT_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || exit 66
[[ "$ADAPTER_RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || exit 67
[[ ! -e "$OUTPUT_DIR" ]] || { echo "merge output already exists" >&2; exit 68; }
for required in "$PROJECT_ROOT/merge_qwen3vl_lora.py" \
  "$MODEL_DIR/config.json" "$ADAPTER_DIR/adapter_config.json" \
  "$ADAPTER_DIR/adapter_model.safetensors" "$PEFT_OVERLAY"; do
  [[ -e "$required" ]] || { echo "missing required path: $required" >&2; exit 69; }
done
AVAILABLE_KIB=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
AVAILABLE_DISK_KIB=$(df -Pk "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
(( AVAILABLE_KIB >= 20 * 1024 * 1024 )) || { echo "less than 20 GiB RAM available" >&2; exit 70; }
(( AVAILABLE_DISK_KIB >= 12 * 1024 * 1024 )) || { echo "less than 12 GiB disk available" >&2; exit 71; }

cd "$PROJECT_ROOT"
env PYTHONPATH="$PEFT_OVERLAY" CUDA_VISIBLE_DEVICES="" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  "$PYTHON" merge_qwen3vl_lora.py \
    --model-dir "$MODEL_DIR" --adapter-dir "$ADAPTER_DIR" \
    --output "$OUTPUT_NAME" \
    --expected-adapter-sha256 "$EXPECTED_ADAPTER_SHA256"
