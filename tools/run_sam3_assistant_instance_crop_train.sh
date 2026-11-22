#!/usr/bin/env bash
set -euo pipefail

: "${LABELAGENT_ROOT:?set LABELAGENT_ROOT}"
: "${SAM3_STAGE0_DATA_ROOT:?set SAM3_STAGE0_DATA_ROOT}"
: "${SAM3_STAGE0_TRAIN_ROOT:?set SAM3_STAGE0_TRAIN_ROOT}"
: "${SAM3_STAGE0_OUTPUT_ROOT:?set SAM3_STAGE0_OUTPUT_ROOT}"
: "${SAM3_STAGE0_BPE_PATH:?set SAM3_STAGE0_BPE_PATH}"
: "${SAM3_STAGE0_CHECKPOINT:?set SAM3_STAGE0_CHECKPOINT}"
: "${SAM3_ASSISTANT_INSTANCE_ACK:?set SAM3_ASSISTANT_INSTANCE_ACK}"

EXPECTED_ACK=I_ACKNOWLEDGE_ASSISTANT_REVIEW_IS_NOT_HUMAN_TRUTH
[[ "$SAM3_ASSISTANT_INSTANCE_ACK" == "$EXPECTED_ACK" ]]
[[ "$SAM3_STAGE0_TRAIN_ROOT" != *test24* ]]
[[ -f "$SAM3_STAGE0_TRAIN_ROOT/_assistant_delegated_pallet_crops.coco.json" ]]
[[ -f "$SAM3_STAGE0_CHECKPOINT" ]]
[[ -f "$SAM3_STAGE0_BPE_PATH" ]]

python_bin=${SAM3_TRAIN_PYTHON:-python}
for port in 18766 18878; do
  curl --fail --silent --show-error --max-time 5 "http://127.0.0.1:${port}/health" >/dev/null
done

free_cards=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '$1 >= 20000 {n++} END {print n+0}')
[[ "$free_cards" -ge 2 ]]
mkdir -p "$SAM3_STAGE0_OUTPUT_ROOT"
export PYTHONPATH="$LABELAGENT_ROOT:$LABELAGENT_ROOT/source/sam3${PYTHONPATH:+:$PYTHONPATH}"
cd "$LABELAGENT_ROOT/source/sam3"
exec "$python_bin" -m sam3.train.train \
  -c configs/sfl_cpd15y/stage0_assistant_instance_crop_epoch.yaml \
  --use-cluster 0 --num-gpus 2
