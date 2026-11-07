#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${LABELAGENT_PAI_ROOT:?Set LABELAGENT_PAI_ROOT}"
: "${LABELAGENT_EXPORT:?Set LABELAGENT_EXPORT to reviewed export}"
: "${LABELAGENT_DATA_ROOT:?Set LABELAGENT_DATA_ROOT to review-data directory}"
export LABELAGENT_PAI_ROOT LABELAGENT_EXPORT LABELAGENT_DATA_ROOT
SIZE="${MODEL_SIZE:-2B}"
[[ "$SIZE" == 2B || "$SIZE" == 8B ]] || { echo 'Pilot supports 2B/8B; 32B training is a separate stage'; exit 1; }
MODE="${1:-smoke}"
[[ "$MODE" == smoke || "$MODE" == pilot ]] || { echo 'Use smoke or pilot'; exit 1; }
PY="$LABELAGENT_PAI_ROOT/venv-qwen/bin/python"
MODEL="$LABELAGENT_PAI_ROOT/models/Qwen3-VL-$SIZE-Instruct"
[[ -f "$MODEL/config.json" ]] || { echo 'Download the pinned model first'; exit 1; }
"$PY" "$SCRIPT_DIR/preflight.py" --out "$LABELAGENT_PAI_ROOT/logs/preflight-before-train.json"
STAMP="$(date -u +%Y%m%dT%H%M%S)_${RANDOM}"
OUT="$LABELAGENT_PAI_ROOT/runs/qwen${SIZE}_${MODE}_${STAMP}"
[[ ! -e "$OUT" ]] || { echo 'Fresh run directory required'; exit 1; }
mkdir -p "$OUT"
"$PY" -m pip freeze > "$OUT/environment.txt"
cp "$LABELAGENT_EXPORT/export_receipt.json" "$OUT/data_receipt.json"
cp "$SCRIPT_DIR/model_pins.json" "$SCRIPT_DIR/upstream_pin.json" "$OUT/"
EXTRA=(--num_train_epochs 2)
LIMIT=21600
if [[ "$MODE" == smoke ]]; then EXTRA=(--max_steps 10); LIMIT=1800; fi
NPROC="${NPROC_PER_NODE:-1}"
PYTHONHASHSEED=3407 timeout --signal=TERM --kill-after=60s "${LIMIT}s" \
  "$LABELAGENT_PAI_ROOT/venv-qwen/bin/torchrun" --standalone --nproc_per_node="$NPROC" \
  "$SCRIPT_DIR/run_qwen.py" \
  --model_name_or_path "$MODEL" --dataset_use labelagent_pilot \
  --output_dir "$OUT/adapter" --bf16 True --tf32 True \
  --lora_enable True --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --tune_mm_vision False --tune_mm_mlp False --tune_mm_llm True \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
  --gradient_checkpointing True --learning_rate 1e-5 --weight_decay 0.01 \
  --model_max_length 4096 --min_pixels 12544 --max_pixels 451584 \
  --data_flatten False --data_packing False --dataloader_num_workers 2 \
  --logging_steps 1 --save_steps 50 --save_total_limit 2 --report_to none \
  --seed 3407 --data_seed 3407 --ddp_find_unused_parameters False \
  "${EXTRA[@]}" 2>&1 | tee "$OUT/train.log"
"$PY" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); assert (p/"adapter/adapter_config.json").exists(), "No LoRA adapter saved"; (p/"TRAIN_COMPLETED.json").write_text(json.dumps({"status":"trainer_completed_not_quality_validated"})); print(p)' "$OUT"
