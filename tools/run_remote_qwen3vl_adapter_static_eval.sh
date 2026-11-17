#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 PROJECT_ROOT OUTPUT_NAME ADAPTER_RUN_NAME" >&2
  exit 64
fi
PROJECT_ROOT=$1
OUTPUT_NAME=$2
ADAPTER_RUN_NAME=$3
SERVICE_ROOT=/home/clw/labelagent_remote_20261110
VLM_SESSION=labelagent-vlm-20261110
GPU_UUID=GPU-f8597845-b2fa-5ca1-6918-816853753818
PYTHON=/home/clw/labelagent_vlm_20261104/venv/bin/python
PEFT_OVERLAY="$PROJECT_ROOT/venv/lib/python3.12/site-packages"
SERVER_DIR=/home/clw/labelagent_remote_20261110/vlm
MODEL_DIR=/home/clw/labelagent_vlm_20261104/models/Qwen3-VL-2B-Instruct
ADAPTER_DIR="$PROJECT_ROOT/$ADAPTER_RUN_NAME/adapter"
REFERENCE_RESULT="$PROJECT_ROOT/generation-eval-train1epoch-production-order-v1/result.json"
OUTPUT_DIR="$PROJECT_ROOT/$OUTPUT_NAME"
TRITON_CACHE_DIR="$PROJECT_ROOT/cache/triton-qwen3vl2b-lora"
TORCHINDUCTOR_CACHE_DIR="$PROJECT_ROOT/cache/torchinductor-qwen3vl2b-lora"

case "$PROJECT_ROOT" in /home/clw/projects/sfl-cpd15y/*) ;; *) exit 65 ;; esac
[[ "$OUTPUT_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || exit 66
[[ "$ADAPTER_RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || exit 67
[[ ! -e "$OUTPUT_DIR" ]] || { echo "evaluation output already exists" >&2; exit 68; }
for required in "$PROJECT_ROOT/evaluate_qwen3vl_merged_generation.py" \
  "$PROJECT_ROOT/evaluate_qwen3vl_lora_generation.py" \
  "$PROJECT_ROOT/evaluate_qwen3vl_lora_teacher_forcing.py" \
  "$PROJECT_ROOT/train_qwen3vl_lora.py" "$PROJECT_ROOT/dataset/dev.jsonl" \
  "$PROJECT_ROOT/dataset/manifest.json" "$PROJECT_ROOT/dataset/contract.json" \
  "$MODEL_DIR/config.json" "$ADAPTER_DIR/adapter_model.safetensors" \
  "$REFERENCE_RESULT" "$SERVER_DIR/local_vlm_server.py" "$PEFT_OVERLAY"; do
  [[ -e "$required" ]] || { echo "missing required path: $required" >&2; exit 69; }
done
tmux has-session -t "$VLM_SESSION" 2>/dev/null || exit 70
VLM_PID=$(pgrep -f "/home/clw/labelagent_vlm_20261104/venv/bin/python -u local_vlm_server.py.*--port 18878" | head -n 1)
[[ -n "$VLM_PID" && $(ps -p "$VLM_PID" -o user= | xargs) == clw ]] || exit 71
VLM_COMMAND=$(ps -p "$VLM_PID" -o args=)
case "$VLM_COMMAND" in
  *"/home/clw/labelagent_vlm_20261104/venv/bin/python -u local_vlm_server.py"*"--port 18878"*) ;;
  *) echo "owned VLM command differs: $VLM_COMMAND" >&2; exit 72 ;;
esac
HEALTH_JSON=$(curl -fsS http://127.0.0.1:18878/health)
INFLIGHT=$(printf '%s' "$HEALTH_JSON" | "$PYTHON" -c \
  'import json,sys; print(json.load(sys.stdin).get("exact_requests_inflight", -1))')
[[ "$INFLIGHT" == 0 ]] || { echo "owned VLM has in-flight requests: $INFLIGHT" >&2; exit 73; }
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

RESTORE_LOG="$PROJECT_ROOT/${OUTPUT_NAME}-restore.log"
restore_service() { bash "$SERVICE_ROOT/start_remote_services.sh" >>"$RESTORE_LOG" 2>&1 || true; }
restore_and_wait() {
  restore_service
  for _ in $(seq 1 240); do
    if curl -fsS http://127.0.0.1:18878/health \
        >"$PROJECT_ROOT/${OUTPUT_NAME}-restored-health.json" 2>/dev/null; then
      return 0
    fi
    if ! tmux has-session -t "$VLM_SESSION" 2>/dev/null; then restore_service; fi
    sleep 1
  done
  return 1
}
cleanup() {
  status=$?
  trap - EXIT
  restore_and_wait || exit 74
  exit "$status"
}
trap cleanup EXIT
tmux kill-session -t "$VLM_SESSION"
for _ in $(seq 1 30); do kill -0 "$VLM_PID" 2>/dev/null || break; sleep 1; done
kill -0 "$VLM_PID" 2>/dev/null && exit 75

cd "$PROJECT_ROOT"
env PYTHONPATH="$PEFT_OVERLAY:$PROJECT_ROOT:$SERVER_DIR" \
  CUDA_VISIBLE_DEVICES="$GPU_UUID" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  TORCHINDUCTOR_CACHE_DIR="$TORCHINDUCTOR_CACHE_DIR" \
  "$PYTHON" evaluate_qwen3vl_merged_generation.py \
    --model-dir "$MODEL_DIR" --adapter-dir "$ADAPTER_DIR" \
    --dataset-dir dataset --contract dataset/contract.json \
    --server-module-dir "$SERVER_DIR" --reference-result "$REFERENCE_RESULT" \
    --cache-implementation static --output "$OUTPUT_NAME"

restore_and_wait
trap - EXIT
echo "owned_vlm_restored=true"
