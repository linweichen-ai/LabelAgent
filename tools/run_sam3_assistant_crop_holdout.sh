#!/usr/bin/env bash
set -euo pipefail

: "${LABELAGENT_ROOT:?set LABELAGENT_ROOT}"
: "${SAM3_ASSISTANT_HOLDOUT_ROOT:?set SAM3_ASSISTANT_HOLDOUT_ROOT}"
: "${SAM3_STAGE0_BPE_PATH:?set SAM3_STAGE0_BPE_PATH}"
: "${SAM3_STAGE0_CHECKPOINT:?set SAM3_STAGE0_CHECKPOINT}"
: "${SAM3_EVAL_CANDIDATE_CHECKPOINT:?set SAM3_EVAL_CANDIDATE_CHECKPOINT}"
: "${SAM3_EVAL_CANDIDATE_SHA256:?set SAM3_EVAL_CANDIDATE_SHA256}"
: "${SAM3_ASSISTANT_HOLDOUT_OUTPUT:?set SAM3_ASSISTANT_HOLDOUT_OUTPUT}"
: "${SAM3_ASSISTANT_HOLDOUT_SHA256:?set SAM3_ASSISTANT_HOLDOUT_SHA256}"

ann_name=_assistant_delegated_pallet_crops_holdout15.coco.json
ann_path="$SAM3_ASSISTANT_HOLDOUT_ROOT/$ann_name"
[[ "$SAM3_ASSISTANT_HOLDOUT_ROOT" != *test24* ]]
[[ "$SAM3_ASSISTANT_HOLDOUT_OUTPUT" != *test24* ]]
[[ -f "$ann_path" ]]
[[ -f "$SAM3_STAGE0_CHECKPOINT" ]]
[[ -f "$SAM3_STAGE0_BPE_PATH" ]]
[[ -f "$SAM3_EVAL_CANDIDATE_CHECKPOINT" ]]
[[ ! -e "$SAM3_ASSISTANT_HOLDOUT_OUTPUT" ]]

actual_ann_sha=$(sha256sum "$ann_path" | awk '{print $1}')
[[ "$actual_ann_sha" == "$SAM3_ASSISTANT_HOLDOUT_SHA256" ]]
actual_candidate_sha=$(sha256sum "$SAM3_EVAL_CANDIDATE_CHECKPOINT" | awk '{print $1}')
[[ "$actual_candidate_sha" == "$SAM3_EVAL_CANDIDATE_SHA256" ]]
candidate_checkpoint=$SAM3_EVAL_CANDIDATE_CHECKPOINT
candidate_checkpoint_sha=$SAM3_EVAL_CANDIDATE_SHA256

for port in 18766 18878; do
  curl --fail --silent --show-error --max-time 5 \
    "http://127.0.0.1:${port}/health" >/dev/null
done
free_cards=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits |
  awk '$1 >= 20000 {n++} END {print n+0}')
[[ "$free_cards" -ge 2 ]]

python_bin=${SAM3_TRAIN_PYTHON:-python}
config=configs/sfl_cpd15y/stage0_assistant_crop_holdout_predictions.yaml
mkdir -p "$SAM3_ASSISTANT_HOLDOUT_OUTPUT"
export SAM3_STAGE0_DATA_ROOT="$SAM3_ASSISTANT_HOLDOUT_ROOT"
export SAM3_STAGE0_TRAIN_ROOT="$SAM3_ASSISTANT_HOLDOUT_ROOT"
export SAM3_STAGE0_VAL_ROOT="$SAM3_ASSISTANT_HOLDOUT_ROOT"
export PYTHONPATH="$LABELAGENT_ROOT:$LABELAGENT_ROOT/source/sam3${PYTHONPATH:+:$PYTHONPATH}"

run_eval() {
  local label=$1
  local checkpoint=$2
  local checkpoint_sha=$3
  export SAM3_STAGE0_OUTPUT_ROOT="$SAM3_ASSISTANT_HOLDOUT_OUTPUT/$label"
  export SAM3_EVAL_CANDIDATE_CHECKPOINT="$checkpoint"
  export SAM3_EVAL_CANDIDATE_SHA256="$checkpoint_sha"
  mkdir -p "$SAM3_STAGE0_OUTPUT_ROOT"
  (
    cd "$LABELAGENT_ROOT/source/sam3"
    "$python_bin" -m sam3.train.train -c "$config" \
      --use-cluster 0 --num-gpus 2
  )
}

run_eval baseline BASELINE BASELINE
run_eval candidate "$candidate_checkpoint" "$candidate_checkpoint_sha"

find_prediction() {
  local root=$1
  local -a matches
  mapfile -t matches < <(find "$root" -type f -name '*.json' -path '*/predictions/*' | sort)
  [[ "${#matches[@]}" -eq 1 ]]
  printf '%s\n' "${matches[0]}"
}

baseline_prediction=$(find_prediction "$SAM3_ASSISTANT_HOLDOUT_OUTPUT/baseline")
candidate_prediction=$(find_prediction "$SAM3_ASSISTANT_HOLDOUT_OUTPUT/candidate")
"$python_bin" "$LABELAGENT_ROOT/tools/compare_sam3_segmentation_predictions.py" \
  --coco "$ann_path" \
  --baseline "$baseline_prediction" \
  --candidate "$candidate_prediction" \
  --output "$SAM3_ASSISTANT_HOLDOUT_OUTPUT/mask_comparison.json" \
  --split assistant_holdout15
