#!/bin/bash
#
# Phase 1 Binary: Initial Judgment Training Pipeline (Binary Classification)
#
# Binary version differences:
# - Only "can" and "cannot" (no "uncertain")
# - Temperature=0 for all inference (greedy decoding)
# - Single trial per question (no sampling, deterministic)
# - 2x2 confusion matrix instead of 3x3
#
# This script runs Phase 1 in modular steps:
# 1. Baseline evaluation (multi-GPU)
# 2. Build training data (reuse existing responses if available)
# 3. Train + Evaluate loop (each epoch as separate process)
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
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --train_batch_size)
            TRAIN_BATCH_SIZE="$2"
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
            echo "Binary Classification Mode: can/cannot only, temperature=0, single trial"
            echo ""
            echo "Options:"
            echo "  --model MODEL          Model name (default: Qwen/Qwen2.5-7B-Instruct)"
            echo "  --num_samples N        Training samples (default: 500)"
            echo "  --val_samples N        Validation samples (default: 1000)"
            echo "  --num_epochs N         Number of epochs (default: 10)"
            echo "  --batch_size N         Inference batch size (default: 16)"
            echo "  --train_batch_size N   Training batch size (default: 4)"
            echo "  --lr RATE              Learning rate (default: 1e-5)"
            echo "  --num_gpus N           Number of GPUs (default: 8)"
            echo "  --output_dir DIR       Output directory (default: experiments/phase1_binary_<model>)"
            echo "  --force                Force re-run all steps (ignore existing outputs)"
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
FORCE="${FORCE:-0}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
LR="${LR:-1e-5}"
NUM_GPUS="${NUM_GPUS:-8}"

# Output directory: auto-generate from model name if not specified
# e.g., "Qwen/Qwen2.5-14B-Instruct" -> "experiments/phase1_binary_Qwen2.5-14B-Instruct"
if [ -z "$OUTPUT_DIR" ]; then
    MODEL_SHORT=$(basename "$MODEL")  # Extract last part: Qwen2.5-14B-Instruct
    OUTPUT_DIR="experiments/phase1_binary_${MODEL_SHORT}"
fi
mkdir -p "$OUTPUT_DIR"

# Print configuration
echo "============== Configuration (Binary Mode) =============="
echo "MODEL:       $MODEL"
echo "NUM_SAMPLES: $NUM_SAMPLES"
echo "VAL_SAMPLES: $VAL_SAMPLES"
echo "NUM_EPOCHS:  $NUM_EPOCHS"
echo "BATCH_SIZE:  $BATCH_SIZE (inference)"
echo "TRAIN_BATCH: $TRAIN_BATCH_SIZE"
echo "LR:          $LR"
echo "NUM_GPUS:    $NUM_GPUS"
echo "OUTPUT_DIR:  $OUTPUT_DIR"
echo "FORCE:       $FORCE"
echo ""
echo "Binary Mode: can/cannot only, temperature=0, single trial"
echo "=========================================================="

# ============== Helper Functions ==============
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Clean results file (only key metrics, no noise)
RESULTS_FILE="$OUTPUT_DIR/results.txt"

# Function to append epoch results to clean results file (binary version: 2x2 matrix)
append_results() {
    local epoch=$1
    local epoch_dir="$OUTPUT_DIR/epoch_$epoch"

    # Read metrics from JSON files
    python3 - "$epoch_dir" "$epoch" >> "$RESULTS_FILE" << 'PYEOF'
import json
import sys
from pathlib import Path

epoch_dir = Path(sys.argv[1])
epoch = int(sys.argv[2])

train_json = epoch_dir / "metrics_train.json"
val_json = epoch_dir / "metrics_validation.json"

print(f"\n{'='*60}")
print(f"{'='*60}")
print(f"Epoch {epoch} Results (Binary)")
print(f"{'='*60}")
print(f"{'='*60}")

if train_json.exists():
    with open(train_json) as f:
        m = json.load(f)
    print(f"\n[TRAIN]")
    print(f"  QA Accuracy: {m.get('qa_accuracy', 0)*100:.1f}%")
    print(f"  Judgment Accuracy: {m.get('exact_match_rate', 0)*100:.1f}%")
    if 'f1' in m:
        print(f"  F1 Score: {m.get('f1', 0)*100:.1f}%")

    # 2x2 Confusion matrix (binary)
    c = m.get("confusion", {})
    print(f"\n  Confusion Matrix (2x2):")
    print(f"                    actual_can  actual_cannot")
    print(f"    predicted_can      {c.get('can_can', 0):5d}         {c.get('can_cannot', 0):5d}")
    print(f"    predicted_cannot   {c.get('cannot_can', 0):5d}         {c.get('cannot_cannot', 0):5d}")

    pred = m.get("pred_counts", {})
    actual = m.get("actual_counts", {})
    print(f"\n  Predicted: can={pred.get('can', 0)}, cannot={pred.get('cannot', 0)}")
    print(f"  Actual:    can={actual.get('can', 0)}, cannot={actual.get('cannot', 0)}")

if val_json.exists():
    with open(val_json) as f:
        m = json.load(f)
    print(f"\n[VALIDATION]")
    print(f"  QA Accuracy: {m.get('qa_accuracy', 0)*100:.1f}%")
    print(f"  Judgment Accuracy: {m.get('exact_match_rate', 0)*100:.1f}%")
    if 'f1' in m:
        print(f"  F1 Score: {m.get('f1', 0)*100:.1f}%")

    # 2x2 Confusion matrix (binary)
    c = m.get("confusion", {})
    print(f"\n  Confusion Matrix (2x2):")
    print(f"                    actual_can  actual_cannot")
    print(f"    predicted_can      {c.get('can_can', 0):5d}         {c.get('can_cannot', 0):5d}")
    print(f"    predicted_cannot   {c.get('cannot_can', 0):5d}         {c.get('cannot_cannot', 0):5d}")

    pred = m.get("pred_counts", {})
    actual = m.get("actual_counts", {})
    print(f"\n  Predicted: can={pred.get('can', 0)}, cannot={pred.get('cannot', 0)}")
    print(f"  Actual:    can={actual.get('can', 0)}, cannot={actual.get('cannot', 0)}")
PYEOF
}

# Initialize clean results file
echo "Phase 1 Training Results (Binary: can/cannot)" > "$RESULTS_FILE"
echo "Model: $MODEL" >> "$RESULTS_FILE"
echo "Training samples: $NUM_SAMPLES, Validation samples: $VAL_SAMPLES" >> "$RESULTS_FILE"
echo "Learning rate: $LR" >> "$RESULTS_FILE"
echo "Mode: Binary (can/cannot), temperature=0, single trial" >> "$RESULTS_FILE"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')" >> "$RESULTS_FILE"

# ============== Step 0: Baseline Evaluation ==============
log "Step 0: Baseline evaluation (before any training) - Binary Mode"

mkdir -p "$OUTPUT_DIR/epoch_0"
if [ ! -f "$OUTPUT_DIR/baseline_eval.json" ] || [ "$FORCE" = "1" ]; then
    # Evaluate on training samples (same samples used for training)
    python scripts/step4_evaluate_binary.py \
        --model "$MODEL" \
        --lora_path none \
        --split train \
        --num_samples "$NUM_SAMPLES" \
        --inference_batch_size "$BATCH_SIZE" \
        --num_gpus "$NUM_GPUS" \
        --output_json "$OUTPUT_DIR/epoch_0/metrics_train.json" \
        2>&1 | tee "$OUTPUT_DIR/baseline_train.log"

    # Evaluate on validation samples
    python scripts/step4_evaluate_binary.py \
        --model "$MODEL" \
        --lora_path none \
        --split validation \
        --num_samples "$VAL_SAMPLES" \
        --inference_batch_size "$BATCH_SIZE" \
        --num_gpus "$NUM_GPUS" \
        --output_json "$OUTPUT_DIR/epoch_0/metrics_validation.json" \
        2>&1 | tee "$OUTPUT_DIR/baseline_val.log"

    echo '{"status": "completed"}' > "$OUTPUT_DIR/baseline_eval.json"
    log "Baseline evaluation completed"
else
    log "Baseline evaluation already completed, skipping"
fi

# Append baseline results to clean results file
append_results 0

# ============== Step 1: Collect QA Responses ==============
# For binary mode, we still collect responses but will use single greedy trial during training
log "Step 1: Collecting QA responses (for training data)"

RESPONSES_FILE="$OUTPUT_DIR/responses.jsonl"
if [ ! -f "$RESPONSES_FILE" ] || [ "$FORCE" = "1" ]; then
    # Collect responses with temperature=1 for diversity (used to build training set)
    # But during training/eval, we'll use temperature=0
    if python scripts/step1_collect_responses.py \
        --model "$MODEL" \
        --split train \
        --num_samples "$NUM_SAMPLES" \
        --num_trials 1 \
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
log "Starting training loop ($NUM_EPOCHS epochs) - Binary Mode"

# Track model path (starts with base model, then continues from previous epoch)
CURRENT_MODEL="$MODEL"

for epoch in $(seq 1 $NUM_EPOCHS); do
    log "====== Epoch $epoch/$NUM_EPOCHS (Binary) ======"

    EPOCH_OUTPUT="$OUTPUT_DIR/epoch_$epoch"
    mkdir -p "$EPOCH_OUTPUT"

    # Skip if already completed (check for config.json as proof of successful save)
    if [ -f "$EPOCH_OUTPUT/config.json" ] && [ "$FORCE" != "1" ]; then
        log "Epoch $epoch already completed, skipping to evaluation"
        CURRENT_MODEL="$EPOCH_OUTPUT"
    else
        # Train one epoch (binary mode)
        log "Training epoch $epoch (binary)..."
        # Use set -o pipefail to catch errors through pipe
        set -o pipefail
        if python scripts/step3_train_epoch_binary.py \
            --model "$CURRENT_MODEL" \
            --input "$TRAINING_DATA" \
            --output_dir "$EPOCH_OUTPUT" \
            --epoch "$epoch" \
            --lr "$LR" \
            --batch_size "$TRAIN_BATCH_SIZE" \
            --inference_batch_size "$BATCH_SIZE" \
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

    # Evaluate after this epoch (always run to see progress) - Binary mode
    # 1. Evaluate on training samples (same samples used for training)
    log "Evaluating epoch $epoch on training samples ($NUM_SAMPLES) - Binary..."
    python scripts/step4_evaluate_binary.py \
        --model "$CURRENT_MODEL" \
        --lora_path none \
        --split train \
        --num_samples "$NUM_SAMPLES" \
        --inference_batch_size "$BATCH_SIZE" \
        --num_gpus "$NUM_GPUS" \
        --output_json "$EPOCH_OUTPUT/metrics_train.json" \
        2>&1 | tee "$EPOCH_OUTPUT/eval_train.log"

    # 2. Evaluate on validation samples
    log "Evaluating epoch $epoch on validation samples ($VAL_SAMPLES) - Binary..."
    python scripts/step4_evaluate_binary.py \
        --model "$CURRENT_MODEL" \
        --lora_path none \
        --split validation \
        --num_samples "$VAL_SAMPLES" \
        --inference_batch_size "$BATCH_SIZE" \
        --num_gpus "$NUM_GPUS" \
        --output_json "$EPOCH_OUTPUT/metrics_validation.json" \
        2>&1 | tee "$EPOCH_OUTPUT/eval_val.log"

    # Append results to clean results file
    append_results "$epoch"

    log "Epoch $epoch evaluation complete"
done

# ============== Final Summary ==============
log "Phase 1 Binary complete!"
log "Final model: $CURRENT_MODEL"
log "Results in: $OUTPUT_DIR"
log "Clean results file: $RESULTS_FILE"

# Create symlink to final model
ln -sf "epoch_$NUM_EPOCHS" "$OUTPUT_DIR/final_model"
log "Symlink created: $OUTPUT_DIR/final_model -> epoch_$NUM_EPOCHS"

# Generate summary table (binary version)
log "Generating summary table (binary)..."
python3 - "$OUTPUT_DIR" "$NUM_EPOCHS" << 'PYTHON_SCRIPT'
import json
import sys
from pathlib import Path

output_dir = sys.argv[1] if len(sys.argv) > 1 else "experiments/phase1_binary"
num_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 10

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
            row["train_f1"] = m.get("f1", 0) * 100

    if val_json.exists():
        with open(val_json) as f:
            m = json.load(f)
            row["val_qa"] = m.get("qa_accuracy", 0) * 100
            row["val_judgment"] = m.get("exact_match_rate", 0) * 100
            row["val_f1"] = m.get("f1", 0) * 100

    if len(row) > 1:  # Has at least some data
        rows.append(row)

# Print table
print("\n" + "=" * 90)
print("PHASE 1 SUMMARY (BINARY: can/cannot)")
print("=" * 90)
print(f"{'Epoch':>6} | {'Train QA':>10} | {'Train Judge':>12} | {'Train F1':>10} | {'Val QA':>10} | {'Val Judge':>12} | {'Val F1':>10}")
print("-" * 90)
for row in rows:
    epoch = row["epoch"]
    train_qa = f"{row.get('train_qa', 0):.1f}%" if 'train_qa' in row else "N/A"
    train_j = f"{row.get('train_judgment', 0):.1f}%" if 'train_judgment' in row else "N/A"
    train_f1 = f"{row.get('train_f1', 0):.1f}%" if 'train_f1' in row else "N/A"
    val_qa = f"{row.get('val_qa', 0):.1f}%" if 'val_qa' in row else "N/A"
    val_j = f"{row.get('val_judgment', 0):.1f}%" if 'val_judgment' in row else "N/A"
    val_f1 = f"{row.get('val_f1', 0):.1f}%" if 'val_f1' in row else "N/A"
    print(f"{epoch:>6} | {train_qa:>10} | {train_j:>12} | {train_f1:>10} | {val_qa:>10} | {val_j:>12} | {val_f1:>10}")
print("=" * 90)

# Save to CSV
csv_path = Path(output_dir) / "summary.csv"
with open(csv_path, "w") as f:
    f.write("epoch,train_qa_acc,train_judgment_acc,train_f1,val_qa_acc,val_judgment_acc,val_f1\n")
    for row in rows:
        f.write(f"{row['epoch']},{row.get('train_qa', '')},{row.get('train_judgment', '')},{row.get('train_f1', '')},{row.get('val_qa', '')},{row.get('val_judgment', '')},{row.get('val_f1', '')}\n")
print(f"\nSummary saved to: {csv_path}")
PYTHON_SCRIPT

log "Done!"
