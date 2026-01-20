#!/bin/bash
# HotpotQA Binary Classification Training

set -e

MODEL=${MODEL:-"Qwen/Qwen2.5-7B-Instruct"}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-5000}
VAL_SAMPLES=${VAL_SAMPLES:-1000}
OUTPUT_DIR=${OUTPUT_DIR:-"experiments/hotpotqa_binary"}
BATCH_SIZE=${BATCH_SIZE:-4}
EPOCHS=${EPOCHS:-3}

echo "========================================"
echo "HotpotQA Binary Classification"
echo "========================================"
echo "Model: $MODEL"
echo "Train samples: $TRAIN_SAMPLES"
echo "Val samples: $VAL_SAMPLES"
echo "Output: $OUTPUT_DIR"
echo "========================================"

cd "$(dirname "$0")/.."

python scripts/run_hotpotqa_binary.py \
    --model "$MODEL" \
    --train_samples "$TRAIN_SAMPLES" \
    --val_samples "$VAL_SAMPLES" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS"
