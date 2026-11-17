#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "usage: $0 PROJECT_ROOT OUTPUT_NAME [MAX_SAMPLES] [MAX_STEPS]" >&2
  exit 64
fi

PROJECT_ROOT=$1
OUTPUT_NAME=$2
MAX_SAMPLES=${3:-8}
MAX_STEPS=${4:-2}
SERVICE_ROOT=/home/clw/labelagent_remote_20261110
VLM_SESSION=labelagent-vlm-20261110
GPU_UUID=GPU-f8597845-b2fa-5ca1-6918-816853753818
PYTHON=/home/clw/labelagent_vlm_20261104/venv/bin/python
PEFT_OVERLAY="$PROJECT_ROOT/venv/lib/python3.12/site-packages"
MODEL_DIR=/home/clw/labelagent_vlm_20261104/models/Qwen3-VL-2B-Instruct
OUTPUT_DIR="$PROJECT_ROOT/$OUTPUT_NAME"

case "$PROJECT_ROOT" in
  /home/clw/projects/sfl-cpd15y/*) ;;
  *) echo "project root is outside the isolated training tree" >&2; exit 65 ;;
esac
if [[ ! "$OUTPUT_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "unsafe output name" >&2
  exit 66
fi
if [[ ! "$MAX_SAMPLES" =~ ^[0-9]+$ || ! "$MAX_STEPS" =~ ^[0-9]+$ \
      || "$MAX_SAMPLES" -lt 1 || "$MAX_SAMPLES" -gt 62 \
      || "$MAX_STEPS" -lt 1 || "$MAX_STEPS" -gt 100 ]]; then
  echo "unsafe training bounds" >&2
  exit 75
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "training output already exists: $OUTPUT_DIR" >&2
  exit 67
fi
for required in \
  "$PROJECT_ROOT/train_qwen3vl_lora.py" \
  "$PROJECT_ROOT/dataset/train.jsonl" \
  "$PROJECT_ROOT/dataset/manifest.json" \
  "$PEFT_OVERLAY" \
  "$MODEL_DIR/config.json"; do
  [[ -e "$required" ]] || { echo "missing required path: $required" >&2; exit 68; }
done
if ! tmux has-session -t "$VLM_SESSION" 2>/dev/null; then
  echo "expected owned VLM tmux session is missing" >&2
  exit 69
fi

VLM_PID=$(pgrep -f "/home/clw/labelagent_vlm_20261104/venv/bin/python -u local_vlm_server.py.*--port 18878" | head -n 1)
[[ -n "$VLM_PID" ]] || { echo "owned VLM process is missing" >&2; exit 70; }
VLM_USER=$(ps -p "$VLM_PID" -o user= | xargs)
[[ "$VLM_USER" == clw ]] || { echo "owned VLM process has unexpected owner: $VLM_USER" >&2; exit 71; }
VLM_COMMAND=$(ps -p "$VLM_PID" -o args=)
case "$VLM_COMMAND" in
  *"/home/clw/labelagent_vlm_20261104/venv/bin/python -u local_vlm_server.py"*"--port 18878"*) ;;
  *) echo "owned VLM command differs: $VLM_COMMAND" >&2; exit 72 ;;
esac
HEALTH_JSON=$(curl -fsS http://127.0.0.1:18878/health)
INFLIGHT=$(printf '%s' "$HEALTH_JSON" | "$PYTHON" -c \
  'import json,sys; print(json.load(sys.stdin).get("exact_requests_inflight", -1))')
[[ "$INFLIGHT" == 0 ]] || { echo "owned VLM has in-flight requests: $INFLIGHT" >&2; exit 76; }

RESTORE_LOG="$PROJECT_ROOT/${OUTPUT_NAME}-restore.log"
restore_service() {
  bash "$SERVICE_ROOT/start_remote_services.sh" >>"$RESTORE_LOG" 2>&1 || true
}
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

echo "owned_vlm_pid=$VLM_PID"
echo "owned_vlm_command=$VLM_COMMAND"
tmux kill-session -t "$VLM_SESSION"
for _ in $(seq 1 30); do
  kill -0 "$VLM_PID" 2>/dev/null || break
  sleep 1
done
if kill -0 "$VLM_PID" 2>/dev/null; then
  echo "owned VLM process did not stop" >&2
  exit 73
fi
echo "owned_vlm_stopped=true"
nvidia-smi --query-gpu=index,uuid,memory.used,memory.free,utilization.gpu --format=csv,noheader

cd "$PROJECT_ROOT"
env \
  PYTHONPATH="$PEFT_OVERLAY" \
  CUDA_VISIBLE_DEVICES="$GPU_UUID" \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false \
  "$PYTHON" train_qwen3vl_lora.py \
    --model-dir "$MODEL_DIR" \
    --dataset-dir dataset \
    --output "$OUTPUT_NAME" \
    --max-samples "$MAX_SAMPLES" \
    --max-steps "$MAX_STEPS" \
    --gradient-accumulation-steps 2 \
    --learning-rate 2e-4
echo "training_command_completed=true"

restore_and_wait
trap - EXIT
echo "owned_vlm_restored=true"
