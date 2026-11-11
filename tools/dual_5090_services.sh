#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${LABELAGENT_5090_CONFIG:-$HOME/projects/labelagent-5090/dual-5090.env}"
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Missing config: $CONFIG_FILE" >&2
  exit 2
fi
# The config is a local, user-owned deployment file and must not contain secrets.
# shellcheck disable=SC1090
source "$CONFIG_FILE"

required=(PROJECT_ROOT SAM_PYTHON VLM_PYTHON SAM_GPU_UUID VLM_GPU_UUID VLM_MODEL_DIR
          SAM_PORT VLM_PORT SAM_LOG VLM_LOG RUN_DIR)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing configuration value: $name" >&2
    exit 2
  fi
done

mkdir -p "$RUN_DIR" "$(dirname "$SAM_LOG")" "$(dirname "$VLM_LOG")"

pid_file() { printf '%s/%s.pid\n' "$RUN_DIR" "$1"; }

is_ours() {
  local service="$1" pidfile pid command
  pidfile="$(pid_file "$service")"
  [[ -s "$pidfile" ]] || return 1
  pid="$(<"$pidfile")"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  command="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
  [[ "$command" == *"$PROJECT_ROOT"* || "$command" == *"source.sam3_service.server"* ]]
}

check_gpu() {
  local python="$1" uuid="$2"
  "$python" "$PROJECT_ROOT/tools/vlm_resources.py" \
    --hardware-profile dedicated-5090 --gpu-uuid "$uuid" --minimum-free-mib 28000 >/dev/null
}

wait_health() {
  local port="$1" limit="$2" pid="$3"
  local url="http://127.0.0.1:$port/health"
  local count
  for ((count=0; count<limit; count++)); do
    if curl --fail --silent "$url" >/dev/null; then
      curl --fail --silent "$url"
      printf '\n'
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "Service exited before health check passed" >&2
      return 1
    fi
    sleep 1
  done
  echo "Timed out waiting for $url" >&2
  return 1
}

start_sam() {
  if is_ours sam; then echo "sam already running"; return 0; fi
  [[ -x "$SAM_PYTHON" && -d "$PROJECT_ROOT/source/sam3" ]] || {
    echo "SAM environment or source is missing" >&2; return 1; }
  compgen -G "$PROJECT_ROOT/source/sam3/weights/*.pt" >/dev/null || {
    echo "SAM checkpoint is missing from source/sam3/weights" >&2; return 1; }
  check_gpu "$SAM_PYTHON" "$SAM_GPU_UUID"
  cd "$PROJECT_ROOT"
  nohup env CUDA_VISIBLE_DEVICES="$SAM_GPU_UUID" SAM3_HOST=127.0.0.1 SAM3_PORT="$SAM_PORT" \
    "$SAM_PYTHON" -u -m source.sam3_service.server >"$SAM_LOG" 2>&1 </dev/null &
  local pid=$!
  printf '%s\n' "$pid" >"$(pid_file sam)"
  wait_health "$SAM_PORT" 120 "$pid"
}

start_vlm() {
  if is_ours vlm; then echo "vlm already running"; return 0; fi
  [[ -x "$VLM_PYTHON" && -f "$VLM_MODEL_DIR/config.json" ]] || {
    echo "VLM environment or model is missing" >&2; return 1; }
  check_gpu "$VLM_PYTHON" "$VLM_GPU_UUID"
  cd "$PROJECT_ROOT"
  nohup env CUDA_VISIBLE_DEVICES="$VLM_GPU_UUID" \
    "$VLM_PYTHON" -u tools/local_vlm_server.py \
      --model-dir "$VLM_MODEL_DIR" --cache-dir "$PROJECT_ROOT/work/cache/vlm" \
      --gpu-uuid "$VLM_GPU_UUID" --model-id labelagent-vlm8b --port "$VLM_PORT" \
      --minimum-free-mib 28000 --context-budget 16384 --quantization nf4 \
      --max-image-tokens 512 --profile dedicated-5090 >"$VLM_LOG" 2>&1 </dev/null &
  local pid=$!
  printf '%s\n' "$pid" >"$(pid_file vlm)"
  wait_health "$VLM_PORT" 600 "$pid"
}

stop_service() {
  local service="$1" pidfile pid
  pidfile="$(pid_file "$service")"
  if ! is_ours "$service"; then
    echo "$service is not running under this deployment"
    return 0
  fi
  pid="$(<"$pidfile")"
  kill -TERM "$pid"
  for _ in {1..30}; do
    if ! kill -0 "$pid" 2>/dev/null; then rm -f "$pidfile"; echo "$service stopped"; return 0; fi
    sleep 1
  done
  echo "$service did not exit after SIGTERM; left running for manual inspection" >&2
  return 1
}

status_service() {
  local service="$1" port="$2"
  if is_ours "$service"; then
    printf '%s running pid=%s ' "$service" "$(<"$(pid_file "$service")")"
    curl --fail --silent "http://127.0.0.1:$port/health" || true
    printf '\n'
  else
    echo "$service not running under this deployment"
  fi
}

action="${1:-status}"
service="${2:-all}"
case "$action:$service" in
  start:sam) start_sam ;;
  start:vlm) start_vlm ;;
  start:all) start_sam; start_vlm ;;
  stop:sam) stop_service sam ;;
  stop:vlm) stop_service vlm ;;
  stop:all) stop_service vlm; stop_service sam ;;
  status:sam) status_service sam "$SAM_PORT" ;;
  status:vlm) status_service vlm "$VLM_PORT" ;;
  status:all) status_service sam "$SAM_PORT"; status_service vlm "$VLM_PORT" ;;
  *) echo "Usage: $0 {start|stop|status} {sam|vlm|all}" >&2; exit 2 ;;
esac
