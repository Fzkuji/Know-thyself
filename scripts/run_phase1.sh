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

# ============== Parse Command Line Arguments ==============
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --val_samples)
            VAL_SAMPLES="$2"
            shift 2
            ;;
        --num_trials)
            NUM_TRIALS="$2"
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL          Model name (default: Qwen/Qwen2.5-7B-Instruct)"
            echo "  --num_samples N        Training samples (default: 500)"
            echo "  --val_samples N        Validation samples (default: 1000)"
            echo "  --num_trials N         Trials per question (default: 10)"
            echo "  --num_epochs N         Number of epochs (default: 10)"
            echo "  --batch_size N         Inference batch size (default: 32)"
            echo "  --lr RATE              Learning rate (default: 1e-5)"
            echo "  --num_gpus N           Number of GPUs (default: 8)"
            echo "  --output_dir DIR       Output directory (default: experiments/phase1)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============== Configuration (with defaults) ==============
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
NUM_SAMPLES="${NUM_SAMPLES:-500}"
VAL_SAMPLES="${VAL_SAMPLES:-1000}"  # validation samples
NUM_TRIALS="${NUM_TRIALS:-10}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-5}"
NUM_GPUS="${NUM_GPUS:-8}"

# Output directory: auto-generate from model name if not specified
# e.g., "Qwen/Qwen2.5-14B-Instruct" -> "experiments/phase1_Qwen2.5-14B-Instruct"
if [ -z "$OUTPUT_DIR" ]; then
    MODEL_SHORT=$(basename "$MODEL")  # Extract last part: Qwen2.5-14B-Instruct
    OUTPUT_DIR="experiments/phase1_${MODEL_SHORT}"
fi
mkdir -p "$OUTPUT_DIR"

# Print configuration
echo "============== Configuration =============="
echo "MODEL:       $MODEL"
echo "NUM_SAMPLES: $NUM_SAMPLES"
echo "VAL_SAMPLES: $VAL_SAMPLES"
echo "NUM_TRIALS:  $NUM_TRIALS"
echo "NUM_EPOCHS:  $NUM_EPOCHS"
echo "BATCH_SIZE:  $BATCH_SIZE"
echo "LR:          $LR"
echo "NUM_GPUS:    $NUM_GPUS"
echo "OUTPUT_DIR:  $OUTPUT_DIR"
echo "==========================================="

# ============== Helper Functions ==============
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# ============== Step 0: Baseline Evaluation ==============
log "Step 0: Baseline evaluation (before any training)"

mkdir -p "$OUTPUT_DIR/epoch_0"
if [ ! -f "$OUTPUT_DIR/baseline_eval.json" ]; then
    # Evaluate on training samples (same samples used for training)
    python scripts/step4_evaluate.py \
        --model "$MODEL" \
        --lora_path none \
        --split train \
        --num_samples "$NUM_SAMPLES" \
        --num_trials "$NUM_TRIALS" \
        --inference_batch_size "$BATCH_SIZE" \
        --num_gpus "$NUM_GPUS" \
        --output_json "$OUTPUT_DIR/epoch_0/metrics_train.json" \
        2>&1 | tee "$OUTPUT_DIR/baseline_train.log"

    # Evaluate on validation samples
    python scripts/step4_evaluate.py \
        --model "$MODEL" \
        --lora_path none \
        --split validation \
        --num_samples "$VAL_SAMPLES" \
        --num_trials "$NUM_TRIALS" \
        --inference_batch_size "$BATCH_SIZE" \
        --num_gpus "$NUM_GPUS" \
        --output_json "$OUTPUT_DIR/epoch_0/metrics_validation.json" \
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
    if python scripts/step1_collect_responses.py \
        --model "$MODEL" \
        --split train \
        --num_samples "$NUM_SAMPLES" \
        --num_trials "$NUM_TRIALS" \
        --inference_batch_size "$BATCH_SIZE" \
        --num_gpus "$NUM_GPUS" \
        --output "$RESPONSES_FILE" \
        2>&1 | tee "$OUTPUT_DIR/step1.log"; then
        log "Response collection completed"
    else
        log "ERROR: Response collection failed!"
        exit 1
    fi
else
    log "Responses already collected, skipping"
fi

# ============== Step 2: Build Training Data ==============
log "Step 2: Building training data"

TRAINING_DATA="$OUTPUT_DIR/training_data.jsonl"
if [ ! -f "$TRAINING_DATA" ]; then
    if python scripts/step2_build_dataset.py \
        --input "$RESPONSES_FILE" \
        --output "$TRAINING_DATA" \
        2>&1 | tee "$OUTPUT_DIR/step2.log"; then
        log "Training data built"
    else
        log "ERROR: Training data build failed!"
        exit 1
    fi
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

    # Skip if already completed (check for config.json as proof of successful save)
    if [ -f "$EPOCH_OUTPUT/config.json" ]; then
        log "Epoch $epoch already completed, skipping to evaluation"
        CURRENT_MODEL="$EPOCH_OUTPUT"
    else
        # Train one epoch
        log "Training epoch $epoch..."
        # Use set -o pipefail to catch errors through pipe
        set -o pipefail
        if python scripts/step3_train_epoch.py \
            --model "$CURRENT_MODEL" \
            --input "$TRAINING_DATA" \
            --output_dir "$EPOCH_OUTPUT" \
            --epoch "$epoch" \
            --lr "$LR" \
            --num_trials "$NUM_TRIALS" \
            --skip_correct \
            --use_realtime_labels \
            2>&1 | tee "$EPOCH_OUTPUT/train.log"; then
            # Verify that model was saved correctly
            if [ ! -f "$EPOCH_OUTPUT/config.json" ]; then
                log "ERROR: Training appeared to succeed but config.json not found!"
                exit 1
            fi
            CURRENT_MODEL="$EPOCH_OUTPUT"
            log "Epoch $epoch training complete"
        else
            log "ERROR: Epoch $epoch training failed!"
            exit 1
        fi
        set +o pipefail
    fi

    # Evaluate after this epoch (always run to see progress)
    # 1. Evaluate on training samples (same samples used for training)
    log "Evaluating epoch $epoch on training samples ($NUM_SAMPLES)..."
    python scripts/step4_evaluate.py \
        --model "$CURRENT_MODEL" \
        --lora_path none \
        --split train \
        --num_samples "$NUM_SAMPLES" \
        --num_trials "$NUM_TRIALS" \
        --inference_batch_size "$BATCH_SIZE" \
        --num_gpus "$NUM_GPUS" \
        --output_json "$EPOCH_OUTPUT/metrics_train.json" \
        2>&1 | tee "$EPOCH_OUTPUT/eval_train.log"

    # 2. Evaluate on validation samples
    log "Evaluating epoch $epoch on validation samples ($VAL_SAMPLES)..."
    python scripts/step4_evaluate.py \
        --model "$CURRENT_MODEL" \
        --lora_path none \
        --split validation \
        --num_samples "$VAL_SAMPLES" \
        --num_trials "$NUM_TRIALS" \
        --inference_batch_size "$BATCH_SIZE" \
        --num_gpus "$NUM_GPUS" \
        --output_json "$EPOCH_OUTPUT/metrics_validation.json" \
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

# Generate summary table
log "Generating summary table..."
python3 << 'PYTHON_SCRIPT'
import json
import os
from pathlib import Path

output_dir = os.environ.get("OUTPUT_DIR", "experiments/phase1")
num_epochs = int(os.environ.get("NUM_EPOCHS", "10"))

# Collect metrics from all epochs
rows = []
for epoch in range(0, num_epochs + 1):
    epoch_dir = Path(output_dir) / f"epoch_{epoch}"
    train_json = epoch_dir / "metrics_train.json"
    val_json = epoch_dir / "metrics_validation.json"

    row = {"epoch": epoch}

    if train_json.exists():
        with open(train_json) as f:
            m = json.load(f)
            row["train_qa"] = m.get("qa_accuracy", 0) * 100
            row["train_judgment"] = m.get("exact_match_rate", 0) * 100

    if val_json.exists():
        with open(val_json) as f:
            m = json.load(f)
            row["val_qa"] = m.get("qa_accuracy", 0) * 100
            row["val_judgment"] = m.get("exact_match_rate", 0) * 100

    if len(row) > 1:  # Has at least some data
        rows.append(row)

# Print table
print("\n" + "=" * 70)
print("PHASE 1 SUMMARY")
print("=" * 70)
print(f"{'Epoch':>6} | {'Train QA':>10} | {'Train Judge':>12} | {'Val QA':>10} | {'Val Judge':>12}")
print("-" * 70)
for row in rows:
    epoch = row["epoch"]
    train_qa = f"{row.get('train_qa', 0):.1f}%" if 'train_qa' in row else "N/A"
    train_j = f"{row.get('train_judgment', 0):.1f}%" if 'train_judgment' in row else "N/A"
    val_qa = f"{row.get('val_qa', 0):.1f}%" if 'val_qa' in row else "N/A"
    val_j = f"{row.get('val_judgment', 0):.1f}%" if 'val_judgment' in row else "N/A"
    print(f"{epoch:>6} | {train_qa:>10} | {train_j:>12} | {val_qa:>10} | {val_j:>12}")
print("=" * 70)

# Save to CSV
csv_path = Path(output_dir) / "summary.csv"
with open(csv_path, "w") as f:
    f.write("epoch,train_qa_acc,train_judgment_acc,val_qa_acc,val_judgment_acc\n")
    for row in rows:
        f.write(f"{row['epoch']},{row.get('train_qa', '')},{row.get('train_judgment', '')},{row.get('val_qa', '')},{row.get('val_judgment', '')}\n")
print(f"\nSummary saved to: {csv_path}")
PYTHON_SCRIPT

log "Done!"
