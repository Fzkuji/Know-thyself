#!/bin/bash
#
# Phase 1: Initial Judgment Training Pipeline
#
# This script runs Phase 1 in modular steps:
# 1. Collect QA responses (multi-GPU)
# 2. Build training data
# 3. Baseline evaluation (multi-GPU)
# 4. Train + Evaluate loop (each epoch as separate process)
#
# Each step is a separate process, so GPU memory is fully released between steps.
#

set -e  # Exit on error

# ============== Configuration ==============
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
TEST_SAMPLES="${TEST_SAMPLES:-100}"
NUM_TRIALS="${NUM_TRIALS:-10}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-1e-5}"
NUM_GPUS="${NUM_GPUS:-8}"

# Output directory
OUTPUT_DIR="${OUTPUT_DIR:-experiments/phase1}"
mkdir -p "$OUTPUT_DIR"

# ============== Helper Functions ==============
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# ============== Step 0: Baseline Evaluation ==============
log "Step 0: Baseline evaluation (before any training)"

if [ ! -f "$OUTPUT_DIR/baseline_eval.json" ]; then
    python scripts/step4_evaluate.py \
        --model "$MODEL" \
        --lora_path none \
        --split train \
        --num_samples "$TEST_SAMPLES" \
        --num_trials "$NUM_TRIALS" \
        --inference_batch_size "$BATCH_SIZE" \
        --num_gpus "$NUM_GPUS" \
        --print_confusion_matrix \
        2>&1 | tee "$OUTPUT_DIR/baseline_train.log"

    python scripts/step4_evaluate.py \
        --model "$MODEL" \
        --lora_path none \
        --split validation \
        --num_samples "$TEST_SAMPLES" \
        --num_trials "$NUM_TRIALS" \
        --inference_batch_size "$BATCH_SIZE" \
        --num_gpus "$NUM_GPUS" \
        --print_confusion_matrix \
        2>&1 | tee "$OUTPUT_DIR/baseline_val.log"

    echo '{"status": "completed"}' > "$OUTPUT_DIR/baseline_eval.json"
    log "Baseline evaluation completed"
else
    log "Baseline evaluation already completed, skipping"
fi

# ============== Step 1: Collect QA Responses ==============
log "Step 1: Collecting QA responses"

RESPONSES_FILE="$OUTPUT_DIR/responses.jsonl"
if [ ! -f "$RESPONSES_FILE" ]; then
    python scripts/step1_collect_responses.py \
        --model "$MODEL" \
        --split train \
        --num_samples "$NUM_SAMPLES" \
        --num_trials "$NUM_TRIALS" \
        --batch_size "$BATCH_SIZE" \
        --num_gpus "$NUM_GPUS" \
        --output "$RESPONSES_FILE" \
        2>&1 | tee "$OUTPUT_DIR/step1.log"
    log "Response collection completed"
else
    log "Responses already collected, skipping"
fi

# ============== Step 2: Build Training Data ==============
log "Step 2: Building training data"

TRAINING_DATA="$OUTPUT_DIR/training_data.jsonl"
if [ ! -f "$TRAINING_DATA" ]; then
    python scripts/step2_build_dataset.py \
        --input "$RESPONSES_FILE" \
        --output "$TRAINING_DATA" \
        2>&1 | tee "$OUTPUT_DIR/step2.log"
    log "Training data built"
else
    log "Training data already exists, skipping"
fi

# ============== Step 3-4: Train + Evaluate Loop ==============
log "Starting training loop ($NUM_EPOCHS epochs)"

# Track model path (starts with base model, then continues from previous epoch)
CURRENT_MODEL="$MODEL"

for epoch in $(seq 1 $NUM_EPOCHS); do
    log "====== Epoch $epoch/$NUM_EPOCHS ======"

    EPOCH_OUTPUT="$OUTPUT_DIR/epoch_$epoch"
    mkdir -p "$EPOCH_OUTPUT"

    # Skip if already completed
    if [ -f "$EPOCH_OUTPUT/training_complete" ]; then
        log "Epoch $epoch already completed, skipping to evaluation"
        CURRENT_MODEL="$EPOCH_OUTPUT"
    else
        # Train one epoch
        log "Training epoch $epoch..."
        python scripts/step3_train_epoch.py \
            --model "$CURRENT_MODEL" \
            --input "$TRAINING_DATA" \
            --output_dir "$EPOCH_OUTPUT" \
            --epoch "$epoch" \
            --lr "$LR" \
            --num_trials "$NUM_TRIALS" \
            --skip_correct \
            --use_realtime_labels \
            2>&1 | tee "$EPOCH_OUTPUT/train.log"

        touch "$EPOCH_OUTPUT/training_complete"
        CURRENT_MODEL="$EPOCH_OUTPUT"
        log "Epoch $epoch training complete"
    fi

    # Evaluate after this epoch (always run to see progress)
    log "Evaluating epoch $epoch (train split)..."
    python scripts/step4_evaluate.py \
        --model "$CURRENT_MODEL" \
        --lora_path none \
        --split train \
        --num_samples "$TEST_SAMPLES" \
        --num_trials "$NUM_TRIALS" \
        --inference_batch_size "$BATCH_SIZE" \
        --num_gpus "$NUM_GPUS" \
        --print_confusion_matrix \
        2>&1 | tee "$EPOCH_OUTPUT/eval_train.log"

    log "Evaluating epoch $epoch (validation split)..."
    python scripts/step4_evaluate.py \
        --model "$CURRENT_MODEL" \
        --lora_path none \
        --split validation \
        --num_samples "$TEST_SAMPLES" \
        --num_trials "$NUM_TRIALS" \
        --inference_batch_size "$BATCH_SIZE" \
        --num_gpus "$NUM_GPUS" \
        --print_confusion_matrix \
        2>&1 | tee "$EPOCH_OUTPUT/eval_val.log"

    log "Epoch $epoch evaluation complete"
done

# ============== Final Summary ==============
log "Phase 1 complete!"
log "Final model: $CURRENT_MODEL"
log "Results in: $OUTPUT_DIR"

# Create symlink to final model
ln -sf "epoch_$NUM_EPOCHS" "$OUTPUT_DIR/final_model"
log "Symlink created: $OUTPUT_DIR/final_model -> epoch_$NUM_EPOCHS"
