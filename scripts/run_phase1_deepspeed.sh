#!/bin/bash
#
# Phase 1: Judgment Training Pipeline with DeepSpeed ZeRO-3
#
# This version uses DeepSpeed for proper multi-GPU training.
# The model is sharded across GPUs using ZeRO-3.
#
# Steps:
# 1. Baseline evaluation (multi-GPU inference)
# 2. Collect QA responses (multi-GPU inference)
# 3. Build training data
# 4. Train + Evaluate loop (DeepSpeed training, multi-GPU evaluation)
#

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

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
        --force)
            FORCE=1
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Phase 1 with DeepSpeed ZeRO-3 for proper multi-GPU training"
            echo ""
            echo "Options:"
            echo "  --model MODEL          Model name (default: Qwen/Qwen2.5-7B-Instruct)"
            echo "  --num_samples N        Training samples (default: 500)"
            echo "  --val_samples N        Validation samples (default: 1000)"
            echo "  --num_trials N         Trials per question (default: 10)"
            echo "  --num_epochs N         Number of epochs (default: 10)"
            echo "  --batch_size N         Inference batch size (default: 16)"
            echo "  --lr RATE              Learning rate (default: 1e-6)"
            echo "  --num_gpus N           Number of GPUs (default: 8)"
            echo "  --output_dir DIR       Output directory"
            echo "  --force                Force re-run all steps"
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
VAL_SAMPLES="${VAL_SAMPLES:-1000}"
FORCE="${FORCE:-0}"
NUM_TRIALS="${NUM_TRIALS:-10}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-1e-6}"
NUM_GPUS="${NUM_GPUS:-8}"

# DeepSpeed config
DS_CONFIG="$PROJECT_DIR/configs/ds_config_zero3.json"

# Output directory
if [ -z "$OUTPUT_DIR" ]; then
    MODEL_SHORT=$(basename "$MODEL")
    OUTPUT_DIR="$PROJECT_DIR/experiments/phase1_ds_${MODEL_SHORT}"
fi
mkdir -p "$OUTPUT_DIR"

# Print configuration
echo "============== Configuration (DeepSpeed ZeRO-3) =============="
echo "MODEL:       $MODEL"
echo "NUM_SAMPLES: $NUM_SAMPLES"
echo "VAL_SAMPLES: $VAL_SAMPLES"
echo "NUM_TRIALS:  $NUM_TRIALS"
echo "NUM_EPOCHS:  $NUM_EPOCHS"
echo "BATCH_SIZE:  $BATCH_SIZE"
echo "LR:          $LR"
echo "NUM_GPUS:    $NUM_GPUS"
echo "OUTPUT_DIR:  $OUTPUT_DIR"
echo "DS_CONFIG:   $DS_CONFIG"
echo "FORCE:       $FORCE"
echo "==============================================================="

# ============== Helper Functions ==============
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# ============== Step 0: Baseline Evaluation ==============
log "Step 0: Baseline evaluation (before any training)"

mkdir -p "$OUTPUT_DIR/epoch_0"
if [ ! -f "$OUTPUT_DIR/baseline_eval.json" ] || [ "$FORCE" = "1" ]; then
    # Evaluate on training samples
    python "$SCRIPT_DIR/step4_evaluate.py" \
        --model "$MODEL" \
        --lora_path none \
        --split train \
        --num_samples "$NUM_SAMPLES" \
        --num_trials "$NUM_TRIALS" \
        --inference_batch_size "$BATCH_SIZE" \
        --num_gpus "$NUM_GPUS" \
        --epoch 0 \
        --output_json "$OUTPUT_DIR/epoch_0/metrics_train.json" \
        2>&1 | tee "$OUTPUT_DIR/baseline_train.log"

    # Evaluate on validation samples
    python "$SCRIPT_DIR/step4_evaluate.py" \
        --model "$MODEL" \
        --lora_path none \
        --split validation \
        --num_samples "$VAL_SAMPLES" \
        --num_trials "$NUM_TRIALS" \
        --inference_batch_size "$BATCH_SIZE" \
        --num_gpus "$NUM_GPUS" \
        --epoch 0 \
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
if [ ! -f "$RESPONSES_FILE" ] || [ "$FORCE" = "1" ]; then
    if python "$SCRIPT_DIR/step1_collect_responses.py" \
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
if [ ! -f "$TRAINING_DATA" ] || [ "$FORCE" = "1" ]; then
    if python "$SCRIPT_DIR/step2_build_dataset.py" \
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
log "Starting training loop ($NUM_EPOCHS epochs) with DeepSpeed"

# Track model path
CURRENT_MODEL="$MODEL"

for epoch in $(seq 1 $NUM_EPOCHS); do
    log "====== Epoch $epoch/$NUM_EPOCHS (DeepSpeed ZeRO-3) ======"

    EPOCH_OUTPUT="$OUTPUT_DIR/epoch_$epoch"
    mkdir -p "$EPOCH_OUTPUT"

    # Skip if already completed
    if [ -f "$EPOCH_OUTPUT/config.json" ] && [ "$FORCE" != "1" ]; then
        log "Epoch $epoch already completed, skipping to evaluation"
        CURRENT_MODEL="$EPOCH_OUTPUT"
    else
        # Train one epoch with DeepSpeed
        log "Training epoch $epoch with DeepSpeed..."

        # Use deepspeed launcher
        if deepspeed --num_gpus="$NUM_GPUS" \
            "$SCRIPT_DIR/step3_train_epoch_deepspeed.py" \
            --model "$CURRENT_MODEL" \
            --input "$TRAINING_DATA" \
            --output_dir "$EPOCH_OUTPUT" \
            --epoch "$epoch" \
            --lr "$LR" \
            --num_trials "$NUM_TRIALS" \
            --deepspeed "$DS_CONFIG" \
            2>&1 | tee "$EPOCH_OUTPUT/train.log"; then

            # Verify model was saved
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
    fi

    # Evaluate after this epoch
    log "Evaluating epoch $epoch on training samples..."
    python "$SCRIPT_DIR/step4_evaluate.py" \
        --model "$CURRENT_MODEL" \
        --lora_path none \
        --split train \
        --num_samples "$NUM_SAMPLES" \
        --num_trials "$NUM_TRIALS" \
        --inference_batch_size "$BATCH_SIZE" \
        --num_gpus "$NUM_GPUS" \
        --epoch "$epoch" \
        --output_json "$EPOCH_OUTPUT/metrics_train.json" \
        2>&1 | tee "$EPOCH_OUTPUT/eval_train.log"

    log "Evaluating epoch $epoch on validation samples..."
    python "$SCRIPT_DIR/step4_evaluate.py" \
        --model "$CURRENT_MODEL" \
        --lora_path none \
        --split validation \
        --num_samples "$VAL_SAMPLES" \
        --num_trials "$NUM_TRIALS" \
        --inference_batch_size "$BATCH_SIZE" \
        --num_gpus "$NUM_GPUS" \
        --epoch "$epoch" \
        --output_json "$EPOCH_OUTPUT/metrics_validation.json" \
        2>&1 | tee "$EPOCH_OUTPUT/eval_val.log"

    log "Epoch $epoch evaluation complete"
done

# ============== Final Summary ==============
log "Phase 1 (DeepSpeed) complete!"
log "Final model: $CURRENT_MODEL"
log "Results in: $OUTPUT_DIR"

# Create symlink to final model
ln -sf "epoch_$NUM_EPOCHS" "$OUTPUT_DIR/final_model"
log "Symlink created: $OUTPUT_DIR/final_model -> epoch_$NUM_EPOCHS"

# Generate summary table
log "Generating summary table..."
python3 - "$OUTPUT_DIR" "$NUM_EPOCHS" << 'PYTHON_SCRIPT'
import json
import sys
from pathlib import Path

output_dir = sys.argv[1] if len(sys.argv) > 1 else "experiments/phase1"
num_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 10

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

    if len(row) > 1:
        rows.append(row)

print("\n" + "=" * 70)
print("PHASE 1 SUMMARY (DeepSpeed ZeRO-3)")
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

csv_path = Path(output_dir) / "summary.csv"
with open(csv_path, "w") as f:
    f.write("epoch,train_qa_acc,train_judgment_acc,val_qa_acc,val_judgment_acc\n")
    for row in rows:
        f.write(f"{row['epoch']},{row.get('train_qa', '')},{row.get('train_judgment', '')},{row.get('val_qa', '')},{row.get('val_judgment', '')}\n")
print(f"\nSummary saved to: {csv_path}")
PYTHON_SCRIPT

log "Done!"
